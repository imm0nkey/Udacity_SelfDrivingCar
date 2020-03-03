import numpy as np


# Defining the sigmoid function for activations
def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1 + np.exp(-x))


# Derivative of the sigmoid function
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


# The learning rate, eta in the weight step equation
learnrate = 0.5
# Input data
x = np.array([1, 2])
# Target
y = np.array(0.5)

# Input to output weights (Initial weights)
w = np.array([0.5, -0.5])

# Calculate one gradient descent step for each weight
# TODO: Calculate output of neural network
# THe neural network output (y-heat)
# nn_output = sigmoid(x[0]*weights[0] + x[1]*weights[1])
nn_output = sigmoid(np.dot(x, w))

# TODO: Calculate error of neural network
# output error (y - y-hat)
error = y - nn_output

# error term (lowercase delta)
# error_term = error * sigmoid_prime(np.dot(x,weights))

# TODO: Calculate change in weights
# del_w = [learnrate * error_term * x[0], learnrate * error_term * x[1]]
# or del_w = learnrate * error_term * x
del_w = learnrate * error * nn_output * (1 - nn_output) * x

print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)
