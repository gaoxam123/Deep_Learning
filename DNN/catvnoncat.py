import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from deepNN import *

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

#index = 10
#plt.imshow(train_x_orig[index])
#print("y = " + str(train_y[0, index]) + " It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
test_x = test_x_flatten / 255

n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
learning_rate = 0.0075

# parameters, costs = two_layer_model(train_x, train_y, layer_dims=(n_x, n_h, n_y), learning_rate=0.0075, iterations=2500, print_cost=True)
# plot_costs(costs, learning_rate)

# print(predict(train_x, train_y, parameters))

# print(predict(test_x, test_y, parameters))

layers_dims = [12288, 20, 7, 5, 1]
parameters, costs = L_layer_model(train_x, train_y, layer_dims=layers_dims, learning_rate=0.0075, iterations=3000, print_cost=True)
plot_costs(costs, learning_rate)

print(predict(train_x, train_y, parameters))

print(predict(test_x, test_y, parameters))

