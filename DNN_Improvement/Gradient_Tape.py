import numpy as np
import tensorflow as tf
from tensorflow import keras

def cofi_cost_function(X, W, b, Y, R, lambda_):
    nm, nu = Y.shape
    J = 0
    for j in range(nu):
        w = W[j, :]
        b_j = b[0, j]
        for i in range(nm):
            x = X[i, :]
            y = Y[i, j]
            r = R[i, j]
            J += r * (np.dot(x, w) + b_j - y) ** 2

    J += lambda_ * (np.sum(np.square(W)) + np.sum(np.square(X)))
    J /= 2

    return J


opt = keras.optimizers.Adam(learning_rate=1e-1)

iterations = 200
lambda_ = 1
X, W, b, num_movies, num_features, num_users = 0, 0, 0, 0, 0, 0
Y, R = 0, 0

with tf.GradientTape() as tape:
    cost_value = cofi_cost_function(X, W, b, Y, R, lambda_)

grads = tape.gradient(cost_value, [X, W, b])
opt.apply_gradients(zip(grads, [X, W, b]))

# def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
#           num_epochs = 1500, minibatch_size = 32, print_cost = True):
    
#     costs = []                                        # To keep track of the cost
#     train_acc = []
#     test_acc = []
    
#     # Initialize your parameters
#     #(1 line)
#     parameters = initialize_parameters()

#     W1 = parameters['W1']
#     b1 = parameters['b1']
#     W2 = parameters['W2']
#     b2 = parameters['b2']
#     W3 = parameters['W3']
#     b3 = parameters['b3']

#     optimizer = tf.keras.optimizers.Adam(learning_rate)
    
#     # The CategoricalAccuracy will track the accuracy for this multiclass problem
#     test_accuracy = tf.keras.metrics.CategoricalAccuracy()
#     train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    
#     dataset = tf.data.Dataset.zip((X_train, Y_train))
#     test_dataset = tf.data.Dataset.zip((X_test, Y_test))
    
#     # We can get the number of elements of a dataset using the cardinality method
#     m = dataset.cardinality().numpy()
    
#     minibatches = dataset.batch(minibatch_size).prefetch(8)
#     test_minibatches = test_dataset.batch(minibatch_size).prefetch(8)
#     #X_train = X_train.batch(minibatch_size, drop_remainder=True).prefetch(8)# <<< extra step    
#     #Y_train = Y_train.batch(minibatch_size, drop_remainder=True).prefetch(8) # loads memory faster 

#     # Do the training loop
#     for epoch in range(num_epochs):

#         epoch_total_loss = 0.
        
#         #We need to reset object to start measuring from 0 the accuracy each epoch
#         train_accuracy.reset_states()
        
#         for (minibatch_X, minibatch_Y) in minibatches:
            
#             with tf.GradientTape() as tape:
#                 # 1. predict
#                 Z3 = forward_propagation(tf.transpose(minibatch_X), parameters)

#                 # 2. loss
#                 minibatch_total_loss = compute_total_loss(Z3, tf.transpose(minibatch_Y))

#             # We accumulate the accuracy of all the batches
#             train_accuracy.update_state(minibatch_Y, tf.transpose(Z3))
            
#             trainable_variables = [W1, b1, W2, b2, W3, b3]
#             grads = tape.gradient(minibatch_total_loss, trainable_variables)
#             optimizer.apply_gradients(zip(grads, trainable_variables))
#             epoch_total_loss += minibatch_total_loss
        
#         # We divide the epoch total loss over the number of samples
#         epoch_total_loss /= m

#         # Print the cost every 10 epochs
#         if print_cost == True and epoch % 10 == 0:
#             print ("Cost after epoch %i: %f" % (epoch, epoch_total_loss))
#             print("Train accuracy:", train_accuracy.result())
            
#             # We evaluate the test set every 10 epochs to avoid computational overhead
#             for (minibatch_X, minibatch_Y) in test_minibatches:
#                 Z3 = forward_propagation(tf.transpose(minibatch_X), parameters)
#                 test_accuracy.update_state(minibatch_Y, tf.transpose(Z3))
#             print("Test_accuracy:", test_accuracy.result())

#             costs.append(epoch_total_loss)
#             train_acc.append(train_accuracy.result())
#             test_acc.append(test_accuracy.result())
#             test_accuracy.reset_states()


#     return parameters, costs, train_acc, test_acc