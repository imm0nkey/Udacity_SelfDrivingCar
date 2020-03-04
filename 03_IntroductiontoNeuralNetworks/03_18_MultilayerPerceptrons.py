import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


# Network size
N_input = 4
N_hidden = 3
N_output = 2

np.random.seed(42)
# Make some fake data
X = np.random.randn(4)

# Number of records and input units
# n_records, n_inputs = features,shape
# Number of hidden units
# n_hidden = 2
# weights_input_to_hidden = np.random.normal(0, n_inputs**-0.5, size=(n_inputs, n_hidden))
weights_input_to_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))
weights_hidden_to_output = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))

# TODO: Make a forward pass through the network
# hidden_inputs = np.dot(inputs, weights_input_to_hidden)
# Same weights and features as above, but swapped the order
# hidden_inputs = np.dot(weights_input_to_hidden, features)
# Will cause "ValueError"
hidden_layer_in = np.dot(X, weights_input_to_hidden)
hidden_layer_out = sigmoid(hidden_layer_in)

print('Hidden-layer Output:')
print(hidden_layer_out)
