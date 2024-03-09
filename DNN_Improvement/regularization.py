import numpy as np

# def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):    
#     grads = {}
#     costs = []                            # to keep track of the cost
#     m = X.shape[1]                        # number of examples
#     layers_dims = [X.shape[0], 20, 3, 1]
    
#     # Initialize parameters dictionary.
#     parameters = initialize_parameters(layers_dims)

#     # Loop (gradient descent)

#     for i in range(0, num_iterations):

#         # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
#         if keep_prob == 1:
#             a3, cache = forward_propagation(X, parameters)
#         elif keep_prob < 1:
#             a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        
#         # Cost function
#         if lambd == 0:
#             cost = compute_cost(a3, Y)
#         else:
#             cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
            
#         # Backward propagation.
#         assert (lambd == 0 or keep_prob == 1)   # it is possible to use both L2 regularization and dropout, 
#                                                 # but this assignment will only explore one at a time
#         if lambd == 0 and keep_prob == 1:
#             grads = backward_propagation(X, Y, cache)
#         elif lambd != 0:
#             grads = backward_propagation_with_regularization(X, Y, cache, lambd)
#         elif keep_prob < 1:
#             grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
        
#         # Update parameters.
#         parameters = update_parameters(parameters, grads, learning_rate)
        
#         # Print the loss every 10000 iterations
#         if print_cost and i % 10000 == 0:
#             print("Cost after iteration {}: {}".format(i, cost))
#         if print_cost and i % 1000 == 0:
#             costs.append(cost)
    
#     # plot the cost
#     plt.plot(costs)
#     plt.ylabel('cost')
#     plt.xlabel('iterations (x1,000)')
#     plt.title("Learning rate =" + str(learning_rate))
#     plt.show()
    
#     return parameters

def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization. See formula (2) above.
    
    Arguments:
    A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model
    
    Returns:
    cost - value of the regularized loss function (formula (2))
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    #cross_entropy_cost = compute_cost(A3, Y) # This gives you the cross-entropy part of the cost
    
    #(≈ 1 lines of code)
    # L2_regularization_cost = 
    # YOUR CODE STARTS HERE
    L2_regularization_cost = (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) * (lambd / (2 * m))
    
    # YOUR CODE ENDS HERE
    
    #cost = cross_entropy_cost + L2_regularization_cost
    
    #return cost

def backward_propagation_with_regularization(X, Y, cache, lambd): 
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    #(≈ 1 lines of code)
    # dW3 = 1./m * np.dot(dZ3, A2.T) + None
    # YOUR CODE STARTS HERE
    dW3 = 1./m * np.dot(dZ3, A2.T) + (lambd / m) * W3
    
    # YOUR CODE ENDS HERE
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    #(≈ 1 lines of code)
    # dW2 = 1./m * np.dot(dZ2, A1.T) + None
    # YOUR CODE STARTS HERE
    dW2 = 1./m * np.dot(dZ2, A1.T) + (lambd / m) * W2
    
    # YOUR CODE ENDS HERE
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    #(≈ 1 lines of code)
    # dW1 = 1./m * np.dot(dZ1, X.T) + None
    # YOUR CODE STARTS HERE
    dW1 = 1./m * np.dot(dZ1, X.T) + (lambd / m) * W1
    
    # YOUR CODE ENDS HERE
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients

# def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
#     np.random.seed(1)
    
#     # retrieve parameters
#     W1 = parameters["W1"]
#     b1 = parameters["b1"]
#     W2 = parameters["W2"]
#     b2 = parameters["b2"]
#     W3 = parameters["W3"]
#     b3 = parameters["b3"]
    
#     # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
#     Z1 = np.dot(W1, X) + b1
#     A1 = relu(Z1)
#     #(≈ 4 lines of code)         # Steps 1-4 below correspond to the Steps 1-4 described above. 
#     # D1 =                                           # Step 1: initialize matrix D1 = np.random.rand(..., ...)
#     # D1 =                                           # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
#     # A1 =                                           # Step 3: shut down some neurons of A1
#     # A1 =                                           # Step 4: scale the value of neurons that haven't been shut down
#     # YOUR CODE STARTS HERE
#     D1 = np.random.rand(A1.shape[0], A1.shape[1])
#     D1 = (D1 < keep_prob).astype('int')
#     A1 = A1 * D1
#     A1 /= keep_prob
    
#     # YOUR CODE ENDS HERE
#     Z2 = np.dot(W2, A1) + b2
#     A2 = relu(Z2)
#     #(≈ 4 lines of code)
#     # D2 =                                           # Step 1: initialize matrix D2 = np.random.rand(..., ...)
#     # D2 =                                           # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
#     # A2 =                                           # Step 3: shut down some neurons of A2
#     # A2 =                                           # Step 4: scale the value of neurons that haven't been shut down
#     # YOUR CODE STARTS HERE
#     D2 = np.random.rand(A2.shape[0], A2.shape[1])
#     D2 = (D2 < keep_prob).astype('int')
#     A2 = A2 * D2
#     A2 /= keep_prob
    
#     # YOUR CODE ENDS HERE
#     Z3 = np.dot(W3, A2) + b3
#     A3 = sigmoid(Z3)
    
#     cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    
#     return A3, cache

def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    #(≈ 2 lines of code)
    # dA2 =                # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
    # dA2 =                # Step 2: Scale the value of neurons that haven't been shut down
    # YOUR CODE STARTS HERE
    dA2 = dA2 * D2
    dA2 /= keep_prob
    
    # YOUR CODE ENDS HERE
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    #(≈ 2 lines of code)
    # dA1 =                # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
    # dA1 =                # Step 2: Scale the value of neurons that haven't been shut down
    # YOUR CODE STARTS HERE
    dA1 = dA1 * D1
    dA1 /= keep_prob
    
    # YOUR CODE ENDS HERE
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients

