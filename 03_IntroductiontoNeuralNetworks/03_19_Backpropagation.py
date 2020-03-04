import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([0.1, -0.3])

# Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
print(hidden_layer_input)
hidden_layer_output = sigmoid(hidden_layer_input)
print('hidden_layer_output')
print(hidden_layer_output)

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
print(output_layer_in)
output = sigmoid(output_layer_in)
print(output)

# Backwards pass
# TODO: Calculate error
error = target - output
print('error')
print(error)

# TODO: Calculate error gradient for output layer
del_err_output = error * output * (1 - output)
print('del_err_output')
print(del_err_output)

# TODO: Calculate error gradient for hidden layer
del_err_hidden = np.dot(del_err_output, weights_hidden_output) * hidden_layer_output * (1 - hidden_layer_output)
print('del_err_hidden')
print(del_err_hidden)

# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate * del_err_output * hidden_layer_output
print('delta_w_h_o')
print(delta_w_h_o)

# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_h = learnrate * del_err_hidden * x[:, None]
# delta_w_i_h = learnrate * del_err_hidden * x.T won't work for 1D array
print('delta_w_i_h')
print(delta_w_i_h)

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)
