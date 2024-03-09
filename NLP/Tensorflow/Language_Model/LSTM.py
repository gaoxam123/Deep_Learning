import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

X, Y, n_values, indices_values, chords = 0#load_music_utils('data/original_metheny.mid')
n_a = 64

n_values = 90
reshaper = Reshape((1, n_values))
LSTM_cell = LSTM(n_a, return_state=True)
densor = Dense(n_values, activation='softmax')

def djmodel(Tx, LSTM_cell, densor, reshaper):
    n_values = densor.units
    n_a = LSTM_cell.units
    
    X = Input(shape=(Tx, n_values))

    a0 = Input(shape=(n_a, ), name='a0')
    c0 = Input(shape=(n_a, ), name='c0')
    a = a0
    c = c0

    outputs = []

    for t in range(Tx):
        x = X[:, t, :]
        x = reshaper(x)
        _, a, c = LSTM_cell(inputs=x, initial_state=[a, c])
        out = densor(a)
        outputs.append(out)

    model = Model(inputs=[X, a0, c0], outputs=outputs)

    return model

model = djmodel(30, LSTM_cell, densor, reshaper)
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))

history = model.fit([X, a0, c0], list(Y), epochs=100, verbose = 0)

def music_inference_model(LSTM_cell, densor, Ty=100):
    n_values = densor.units
    n_a = LSTM_cell.units

    x0 = Input(shape=(1, n_values))
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    outputs = []

    for t in range(Ty):
        _, a, c = LSTM_cell(inputs=x, initial_state=[a, c])
        out = densor(a)
        outputs.append(a)

        x = tf.math.argmax(out, axis=-1)
        x = tf.one_hot(x, depth=n_values)

        x = RepeatVector(1)(x)

    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)

    return inference_model

inference_model = music_inference_model(LSTM_cell, densor, Ty = 50)

x_initializer = np.zeros((1, 1, n_values))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

def predict_and_sample(inference_model, x_initializer=x_initializer, a_initializer=a_initializer, c_initializer=c_initializer):
    n_values = x_initializer.shape[2]

    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    indices = tf.math.argmax(pred, axis=-1)
    results = to_categorical(indices, num_classes=n_values)

    return results, indices

results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)