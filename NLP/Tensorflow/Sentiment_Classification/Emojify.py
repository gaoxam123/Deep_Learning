import numpy as np
from emo_utils import *
import matplotlib.pyplot as plt

X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')

maxLen = len(max(X_train, key=lambda x: len(x.split())).split()) # key: criterias for outputing the maximum

Y_oh_train = convert_to_one_hot(Y_train, 5)
Y_oh_test = convert_to_one_hot(Y_test, 5)

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt') # using GloVe for embeddings

def sentence_to_avg(sentence, word_to_vec_map):
    any_word = list(word_to_vec_map.keys())[0]
    words = sentence.lower().split()

    avg = np.zeros(shape=word_to_vec_map[any_word].shape)

    count = 0

    for w in words:
        if w in word_to_vec_map:
            avg += word_to_vec_map[w]
            count += 1

    if count > 0:
        avg /= count

    return avg

def model(X, Y, word_to_vec_map, learning_rate=0.01, num_iterations=400):
    any_word = list(word_to_vec_map.keys())[0]

    m = Y.shape[0]
    n_y = len(np.unique(Y))
    n_h = word_to_vec_map[any_word].shape[0]

    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y, 1))

    Y_oh = convert_to_one_hot(Y)

    for t in range(num_iterations):
        cost = 0
        dW = 0
        db = 0

        for i in range(m):
            avg = sentence_to_avg(X[i], word_to_vec_map)
            z = np.dot(W, avg) + b
            a = softmax(z)

            cost += -np.sum(Y_oh[i] * np.log(a))

            dz = a - Y_oh[i]
            dW = np.dot(dz, avg.T)
            db = dz       

            W -= learning_rate * dW
            db -= learning_rate * db

        assert type(cost) == np.float64
        assert cost.shape == (), "incorrect implementation of cost"

        if t % 100 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
            pred = predict(X, Y, W, b, word_to_vec_map) #predict is defined in emo_utils.py

    return pred, W, b

pred, W, b = model(X_train, Y_train, word_to_vec_map)

