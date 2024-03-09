import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from utils import translate_sentence, bleu
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

german = Field(tokenize=tokenize_ger, lower=True, init_token='<sos>', eos_token='<eos>')
english = Field(tokenize=tokenize_eng, lower=True, init_token='<sos>', eos_token='<eos>')

train_data, val_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                  fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

class Transformer(nn.Module):
    def __init__(self, embed_size, source_vocab_size, target_vocab_size, src_pad_idx, trg_pad_idx, heads, num_encoder_layers, num_decoder_layers, dropout_p, max_len, device):
        super().__init__()
        self.src_word_embedding = nn.Embedding(source_vocab_size, embed_size)
        self.src_position_encoding = nn.Embedding(max_len, embed_size)
        self.trg_word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.trg_position_encoding = nn.Embedding(max_len, embed_size)

        self.device = device
        self.transformer = nn.Transformer(embed_size, heads, num_encoder_layers, num_decoder_layers)
        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout_p)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        return src_mask.to(self.device)

    def forward(self, source, target):
        src_seq_len, N = source.shape
        trg_seq_len, N = target.shape
        src_positions = torch.arange(src_seq_len).repeat(N).reshape(N, src_seq_len).to(self.device)
        trg_positions = torch.arange(trg_seq_len).repeat(N).reshape(N, trg_seq_len).to(self.device)

        embed_src = self.dropout(self.src_word_embedding(source) + self.src_position_encoding(src_positions))
        embed_trg = self.dropout(self.trg_word_embedding(target) + self.trg_position_encoding(trg_positions))

        src_padding_mask = self.make_src_mask(source)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len).to(self.device)

        out = self.transformer(embed_src, embed_trg, src_padding_mask=src_padding_mask, trg_mask=trg_mask)

        return self.fc_out(out)
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 5
learning_rate = 3e-4
batch_size = 32

src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embed_size = 512
heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
dropout_p = 0.1
max_len = 100
src_pad_idx = english.vocab.stoi["<pad>"]

train_iter, val_iter, test_iter = BucketIterator.splits((train_data, val_data, test_data), batch_size=batch_size, sort_within_batch=True, sort_key=lambda x: len(x.src), device=device)

model = Transformer(embed_size, src_vocab_size, trg_vocab_size, src_pad_idx, None, heads, num_encoder_layers, num_decoder_layers, dropout_p, max_len, device).to(device)
optimizer = optim.Adam(model.parameters(), learning_rate)
pad_idx = english.vocab.stoi["<pad>"]
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

sentence = "du bist dumm"

for epoch in range(num_epochs):
    model.eval()
    translated_sentence = translate_sentence(model, sentence, german, english, device, max_len)
    print(f"{translated_sentence}")

    model.train()

    for batch_idx, batch in enumerate(train_iter):
        source, target = batch.src.to(device), batch.trg.to(device)

        output = model(source, target[:-1])
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = loss_fn(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

score = bleu(test_data, model, german, english, device)
print(f"bleu score: {score * 100}")