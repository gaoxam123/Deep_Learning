import torch
import torch.nn as nn
import torch.optim as optim
import string
import random
import sys
import unidecode
from torch.utils.tensorboard import SummayWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

all_characters = string.printable
n_characters = len(all_characters)

file = unidecode.unidecode(open('names.txt').read())

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        out = self.embed(x)
        out, (hidden, cell) = self.rnn(out.unsqueeze(1), (hidden, cell))
        out = self.fc(out.reshape(out.shape[0], -1))

        return out, hidden, cell
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        return hidden, cell
    
class Generator(nn.Module):
    def __init__(self):
        self.chunk_len = 250
        self.num_epochs = 5000
        self.batch_size = 1
        self.print_every = 50
        self.hidden_size = 256
        self.num_layers = 2
        self.learning_rate = 0.003

    def char_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = all_characters.index(string[c])

        return tensor

    def get_random_batch(self):
        start_idx = random.randint(0, len(file) - self.chunk_len)
        end_idx = start_idx + self.chunk_len + 1
        text_string = file[start_idx:end_idx]
        text_input = torch.zeros(self.batch_size, self.chunk_len)
        text_target = torch.zeros(self.batch_size, self.chunk_len)

        for i in range(self.batch_size):
            text_input[i, :] = self.char_tensor(text_string[:-1])
            text_target[i, :] = self.char_tensor(text_string[1:])

        return text_input.long(), text_target.long()

    def generate(self, initial_string='A', prediction_len=100, temperature=0.85):
        hidden, cell = self.rnn.init_hidden(self.batch_size) #########
        initial_input = self.char_tensor(initial_string)
        predicted = initial_string

        for p in range(len(initial_string) - 1):
            out, hidden, cell = self.rnn(initial_input[p].view(1).to(device), hidden, cell) #######

        last_char = initial_input[-1]

        for p in range(prediction_len):
            out, hidden, cell = self.rnn(last_char.view(1).to(device), hidden, cell) #########
            output_dist = out.data.view(-1).div(temperature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]
            predicted_char = all_characters[top_char]
            predicted += predicted_char
            last_char = self.char_tensor(predicted_char)

        return predicted

    def train(self):
        self.rnn = RNN(n_characters, self.hidden_size, self.num_layers, n_characters).to(device)
        optimizer = torch.optim.Adam(self.rnn.parameters(), self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(self.num_epochs):
            input, target = self.get_random_batch()
            input, target = input.to(device), target.to(device)

            hidden, cell = self.rnn.init_hidden(self.batch_size)

            optimizer.zero_grad()
            loss = 0

            for c in range(self.chunk_len):
                output, hidden, cell = self.rnn(input[:, c], hidden, cell)
                loss += loss_fn(output, target[:, c])

            loss.backward()
            optimizer.step()
            loss = loss.item() / self.chunk_len

            if epoch % self.print_every == 0:
                print(f"loss: {loss} \n")

gen_names = Generator()
gen_names.train()
print(gen_names.generate('albe'))