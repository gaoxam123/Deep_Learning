import numpy as np
import matplotlib.pyplot as plt

def forward_propagation(x, theta):
    J = theta * x

    return J

def backward_propagation(x, theta):
    dtheta = x

    return dtheta

def gradient_check(x, theta, epsilon=1e-7, print_msg=False):
    theta_plus = theta + epsilon
    theta_minus = theta - epsilon
    J_plus = theta_plus * x
    J_minus = theta_minus * x
    gradapprox = (J_plus - J_minus) / (2 * epsilon)
    grad = x
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator
    
    if print_msg:
        if difference > 2e-7:
            print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
        else:
            print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference

# def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7, print_msg=False):
#     parameters_values, _ = dictionary_to_vector(parameters)
    
#     grad = gradients_to_vector(gradients)
#     num_parameters = parameters_values.shape[0]
#     J_plus = np.zeros((num_parameters, 1))
#     J_minus = np.zeros((num_parameters, 1))
#     gradapprox = np.zeros((num_parameters, 1))
    
#     # Compute gradapprox
#     for i in range(num_parameters):
#         theta_plus = np.copy(parameters_values)
#         theta_plus[i] = theta_plus[i] + epsilon
#         J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_plus))
#         theta_minus = np.copy(parameters_values)
#         theta_minus[i] = theta_minus[i] - epsilon
#         J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_minus))
#         gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

#     numerator = np.linalg.norm(grad - gradapprox)
#     denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
#     difference = numerator / denominator
    
#     # YOUR CODE ENDS HERE
#     if print_msg:
#         if difference > 2e-7:
#             print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
#         else:
#             print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

#     return difference