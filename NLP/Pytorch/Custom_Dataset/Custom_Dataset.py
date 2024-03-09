import torch
import torchtext
import spacy
from torchtext.data import Field, TabularDataset, BucketIterator

#python -m spacy download en
spacy_en = spacy.load('en')

def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

# tokenize = lambda x: x.split()

quote = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
score = Field(sequential=False, use_vocab=False)

fields = {'quote': ('q', quote), 'score': ('s', score)}
# batch.q, batch.s

train_data, test_data = TabularDataset.splits(path='data', train='train.json', test='test.json', format='json', fields=fields)

# print(train_data[0].__dict__.keys()) # q, s
# print(train_data[0].__dict__.values()) # words in the sentence

quote.build_vocab(train_data, max_size=10000, min_freq=1, vectors='glove.6B.100d')

train_iterator, test_iterator = BucketIterator.splits((train_data, test_data), batch_sizes=2, device='cuda')

for batch in train_iterator:
    print(batch.q, batch.s)