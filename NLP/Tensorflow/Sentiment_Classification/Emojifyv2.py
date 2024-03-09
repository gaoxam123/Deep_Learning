import numpy as np
import tensorflow as tf
np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from emo_utils import *
np.random.seed(1)

def sentence_to_indices(X, word_to_index, max_len): # I love u -> [123, 456, 789, 0, 0]
    m = X.shape[0]
    X_indices = np.zeros(shape=(m, max_len))

    for i in range(m):
        sentence_words = X[i].lower().split()
        j = 0
        for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j += 1

    return X_indices

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_size = len(word_to_index) + 1
    any_word = list(word_to_vec_map.keys())[0]
    emb_dim = word_to_vec_map[any_word].shape[0]

    emb_matrix = np.zeros((vocab_size, emb_dim))

    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]

    embedding_layer = Embedding(input_dim=vocab_size, output_dim=emb_dim, trainable=False)

    embedding_layer.build((None, ))
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer

def Emojifyv2(input_shape, word_to_vec_map, word_to_index):
    sentence_indices = Input(shape=(input_shape, ), dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    X = embedding_layer(sentence_indices)
    X = LSTM(units=128, return_sequences=True)(X)
    X = Dropout(0.5)(X)
    X = LSTM(units=128, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(units=5)(X)
    X = Activation('softmax')(X)

    model = Model(inputs=sentence_indices, outputs=X)
    
    return model

X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')

maxLen = len(max(X_train, key=lambda x: len(x.split())).split()) # key: criterias for outputing the maximum

Y_oh_train = convert_to_one_hot(Y_train, 5)
Y_oh_test = convert_to_one_hot(Y_test, 5)

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt') # using GloVe for embeddings

X_train_indices = sentence_to_indices(X_train, word_to_index, max_len=5)
Y_train_oh = convert_to_one_hot(Y_train, 5)

model = Emojifyv2((maxLen, ), word_to_vec_map, word_to_index)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train_indices, Y_train_oh, epochs=50, batch_size=32, shuffle=True)

X_test_indices = sentence_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = 5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)

# mislabelled examples
C = 5
y_test_oh = np.eye(C)[Y_test.reshape(-1)]
X_test_indices = sentence_to_indices(X_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)
for i in range(len(X_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    if(num != Y_test[i]):
        print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num).strip())

