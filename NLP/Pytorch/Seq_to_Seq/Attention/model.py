import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np 
import spacy
import random
from tqdm.auto import tqdm
from NLP.Pytorch.Seq_to_Seq.Basic.utils import *

spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')

def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

german = Field(tokenize=tokenizer_ger, lower=True, init_token='<sos>', eos_token='<sos>')
english = Field(tokenize=tokenizer_eng, lower=True, init_token='<sos>', eos_token='<eos>')

train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                         fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, p):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embed = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=p, bidirectional=True)
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        # x: L x N
        embedding = self.dropout(self.embed(x))
        # embedding: L x N x embed_size
        encoder_states, hidden, cell = self.rnn(embedding, hidden, cell)
        hidden = self.fc_hidden(torch.cat((hidden[0:2], hidden[2:4]), dim=2))
        # hidden: 2 x N x hidden_size
        cell = self.fc_cell(torch.cat((cell[0:2], cell[2:4]), dim=2))

        return encoder_states, hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, output_size, p):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embed = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(hidden_size * 2 + embed_size, hidden_size, num_layers, dropout=p)
        self.energy = nn.Linear(hidden_size * 3, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, encoder_states, hidden, cell):
        # x: N but we want 1 x N
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embed(x))
        # embedding: 1 x N x embed_size

        sequence_len = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_len, 1, 1)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder), dim=2)))
        attention = self.softmax(energy)
        # seq_len, N, 1
        attention = attention.permute(1, 2, 0) # -> N x 1 x seq_len
        encoder_states = encoder_states.permute(1, 0, 2) # N x seq_len x hidden_size * 2

        context = torch.bmm(attention, encoder_states).permute(1, 0, 2)
        rnn_input = torch.cat((context, embedding), dim=2)

        out, hidden, cell = self.rnn(rnn_input, hidden, cell)
        # out: 1 x N x hidden_size
        pred = self.fc(out)
        # pred: 1 x N x output_size
        return pred.squeeze(0), hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_len = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_len).to('cuda')

        encoder_states, hidden, cell = self.encoder(source)

        # grab start token
        x = target[0]
        for t in range(1, target_len):
            out, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            outputs[t] = out
            # out: N x vocab_len
            best_guess = out.argmax(dim=1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs
    
num_epochs = 20
learning_rate = 0.001
batch_size = 64

device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_size_encoder = len(english.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embed_size = 300
decoder_embed_size = 300
hidden_size = 1024
num_layers = 2
enc_p = 0.5
dec_p = 0.5

train_iterator, val_iterator, test_iterator = BucketIterator.splits((train_data, validation_data, test_data),
                                                                    batch_size=batch_size,
                                                                    sort_within_batch=True,
                                                                    sort_key = lambda x: len(x.src),
                                                                    device=device)

encoder = Encoder(input_size_encoder, encoder_embed_size, hidden_size, num_layers, enc_p).to(device)
decoder = Decoder(input_size_decoder, decoder_embed_size, hidden_size, num_layers, output_size, dec_p).to(device)
model = Seq2Seq(encoder, decoder).to(device)

pad_idx = english.vocab.stoi['<pad>']
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

for epoch in tqdm(range(num_epochs)):
    for batch_idx, batch in enumerate(tqdm(train_iterator)):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)
        
        output = model(inp_data, target)
        # output: trg_len, batch_size, output_size
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)
        loss = loss_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()