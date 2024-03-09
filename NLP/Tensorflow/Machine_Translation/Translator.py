import numpy as np
import tensorflow as tf
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply, RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Model, load_model
import keras.backend as K

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from Translator_utils import *
import matplotlib.pyplot as plt

m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
# print("X.shape:", X.shape)
# print("Y.shape:", Y.shape)
# print("Xoh.shape:", Xoh.shape)
# print("Yoh.shape:", Yoh.shape)

repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(units=10, activation='tanh')
densor2 = Dense(units=1, activation='relu')
activator = Activation(activation=softmax, name='attention_weights')
dotor = Dot(axes=1)

def one_step_attention(a, s_prev):
    s_prev = repeator(s_prev)
    concat = concatenator([a, s_prev])
    e = densor1(concat)
    energies = densor2(e)
    alphas = activator(energies)
    context = dotor([alphas, a])

    return context

n_a, n_s = 32, 64
post_activation_LSTM_cell = LSTM(n_s, return_state=True)
output_layer = Dense(len(machine_vocab), activation=softmax)

def modelf(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s, ), name='s0')
    c0 = Input(shape=(n_s, ), name='c0')

    s = s0
    c = c0
    outputs = []

    a = Bidirectional(LSTM(units=n_a, return_sequences=True))(X) # a is the sequence of hidden states

    for t in range(Ty):
        context = one_step_attention(a, s)
        _, s, c = post_activation_LSTM_cell(inputs=context, initial_state=[s, c]) # s, c are next hidden and cell state
        out = output_layer(s)
        outputs.append(out)

    model = Model(inputs=[X, s0, c0], outputs=outputs)

    return model

model = modelf(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))

opt = Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0, 1))

model.fit([Xoh, s0, c0], outputs, epochs=10, batch_size=100)