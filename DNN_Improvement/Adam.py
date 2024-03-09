import numpy as np
import math

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    inc = mini_batch_size

    num_complete_minibatches = math.floor(m / mini_batch_size) 
    for k in range (0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, int(m / mini_batch_size) * mini_batch_size:]
            mini_batch_Y = shuffled_Y[:, int(m / mini_batch_size) * mini_batch_size:]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

    return mini_batches

def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros((parameters["W" + str(l)].shape))
        v["db" + str(l)] = np.zeros((parameters["b" + str(l)].shape))

    return v

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2

    for l in range(1, L + 1):
        v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]
        v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l)]

        parameters["W" + str(l)] -= learning_rate * v["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * v["db" + str(l)]

    return parameters

def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros((parameters["W" + str(l)].shape))
        v["db" + str(l)] = np.zeros((parameters["b" + str(l)].shape))

        s["dW" + str(l)] = np.zeros((parameters["W" + str(l)].shape))
        s["db" + str(l)] = np.zeros((parameters["b" + str(l)].shape))

    return v, s

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(1, L + 1):
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] +  (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] +  (1 - beta1) * grads["db" + str(l)]

        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1 ** t)
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1 ** t)

        s["dW" + str(l)] = beta2 * s["dW" + str(l)] +  (1 - beta2) * np.square(grads["dW" + str(l)])
        s["db" + str(l)] = beta2 * s["db" + str(l)] +  (1 - beta2) * np.square(grads["db" + str(l)])

        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2 ** t)
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2 ** t)


        parameters["W" + str(l)] -= learning_rate * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        parameters["b" + str(l)] -= learning_rate * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)

    return parameters, v, s, v_corrected, s_corrected

# def model(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
#           beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 5000, print_cost = True):
#     L = len(layers_dims)             # number of layers in the neural networks
#     costs = []                       # to keep track of the cost
#     t = 0                            # initializing the counter required for Adam update
#     seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
#     m = X.shape[1]                   # number of training examples
    
#     # Initialize parameters
#     parameters = initialize_parameters(layers_dims)

#     # Initialize the optimizer
#     if optimizer == "gd":
#         pass # no initialization required for gradient descent
#     elif optimizer == "momentum":
#         v = initialize_velocity(parameters)
#     elif optimizer == "adam":
#         v, s = initialize_adam(parameters)
    
#     # Optimization loop
#     for i in range(num_epochs):
        
#         # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
#         seed = seed + 1
#         minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
#         cost_total = 0
        
#         for minibatch in minibatches:

#             # Select a minibatch
#             (minibatch_X, minibatch_Y) = minibatch

#             # Forward propagation
#             a3, caches = forward_propagation(minibatch_X, parameters)

#             # Compute cost and add to the cost total
#             cost_total += compute_cost(a3, minibatch_Y)

#             # Backward propagation
#             grads = backward_propagation(minibatch_X, minibatch_Y, caches)

#             # Update parameters
#             if optimizer == "gd":
#                 parameters = update_parameters_with_gd(parameters, grads, learning_rate)
#             elif optimizer == "momentum":
#                 parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
#             elif optimizer == "adam":
#                 t = t + 1 # Adam counter
#                 parameters, v, s, _, _ = update_parameters_with_adam(parameters, grads, v, s,
#                                                                t, learning_rate, beta1, beta2,  epsilon)
#         cost_avg = cost_total / m
        
#         # Print the cost every 1000 epoch
#         if print_cost and i % 1000 == 0:
#             print ("Cost after epoch %i: %f" %(i, cost_avg))
#         if print_cost and i % 100 == 0:
#             costs.append(cost_avg)
                
#     # plot the cost
#     plt.plot(costs)
#     plt.ylabel('cost')
#     plt.xlabel('epochs (per 100)')
#     plt.title("Learning rate = " + str(learning_rate))
#     plt.show()

#     return parameters

# def model(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
#           beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 5000, print_cost = True, decay=None, decay_rate=1):
#     L = len(layers_dims)             # number of layers in the neural networks
#     costs = []                       # to keep track of the cost
#     t = 0                            # initializing the counter required for Adam update
#     seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
#     m = X.shape[1]                   # number of training examples
#     lr_rates = []
#     learning_rate0 = learning_rate   # the original learning rate
    
#     # Initialize parameters
#     parameters = initialize_parameters(layers_dims)

#     # Initialize the optimizer
#     if optimizer == "gd":
#         pass # no initialization required for gradient descent
#     elif optimizer == "momentum":
#         v = initialize_velocity(parameters)
#     elif optimizer == "adam":
#         v, s = initialize_adam(parameters)
    
#     # Optimization loop
#     for i in range(num_epochs):
        
#         # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
#         seed = seed + 1
#         minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
#         cost_total = 0
        
#         for minibatch in minibatches:

#             # Select a minibatch
#             (minibatch_X, minibatch_Y) = minibatch

#             # Forward propagation
#             a3, caches = forward_propagation(minibatch_X, parameters)

#             # Compute cost and add to the cost total
#             cost_total += compute_cost(a3, minibatch_Y)

#             # Backward propagation
#             grads = backward_propagation(minibatch_X, minibatch_Y, caches)

#             # Update parameters
#             if optimizer == "gd":
#                 parameters = update_parameters_with_gd(parameters, grads, learning_rate)
#             elif optimizer == "momentum":
#                 parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
#             elif optimizer == "adam":
#                 t = t + 1 # Adam counter
#                 parameters, v, s, _, _ = update_parameters_with_adam(parameters, grads, v, s,
#                                                                t, learning_rate, beta1, beta2,  epsilon)
#         cost_avg = cost_total / m
#         if decay:
#             learning_rate = decay(learning_rate0, i, decay_rate)
#         # Print the cost every 1000 epoch
#         if print_cost and i % 1000 == 0:
#             print ("Cost after epoch %i: %f" %(i, cost_avg))
#             if decay:
#                 print("learning rate after epoch %i: %f"%(i, learning_rate))
#         if print_cost and i % 100 == 0:
#             costs.append(cost_avg)
                
#     # plot the cost
#     plt.plot(costs)
#     plt.ylabel('cost')
#     plt.xlabel('epochs (per 100)')
#     plt.title("Learning rate = " + str(learning_rate))
#     plt.show()

#     return parameters

def update_lr(learning_rate0, epoch_num, decay_rate):
    learning_rate = 1 / (1 + decay_rate * epoch_num) * learning_rate0

    return learning_rate

def schedule_lr_decay(learning_rate0, epoch_num, decay_rate, time_interval=1000):
    learning_rate = 1 / (1 + decay_rate * math.floor(epoch_num / time_interval)) * learning_rate0

    return learning_rate

