import numpy as np


"""
You need to change the Add() class below.
"""
"""
Bonus Challenge!

Write your code in Add (scroll down).
"""
"""
Modify Linear#forward so that it linearly transforms
input matrices, weights matrices and a bias vector to
an output.
"""
"""
Fix the Sigmoid class so that it computes the sigmoid function
on the forward pass!

Scroll down to get started.
"""
"""
Implement the backward method of the Sigmoid node.
"""


class Node(object):
    """
    Base class for nodes in the network.

    Arguments:

        'inbound_nodes': A list of nodes with edges into this node.
    """
    def __init__(self, inbound_nodes=[]):
        """
        Node's constructor (runs when the object is instantiated). Sets
        properties that all nodes need.
        """
        # Node(s) from which this Node receives values
        # A list of nodes with edges into this nodes
        self.inbound_nodes = inbound_nodes
        # Node(s) to which this Node passes values
        # A list of nodes that this node outputs to.
        self.outbound_nodes = []
        # A calculated value
        # The eventual value of this node. Set by running
        # the forward() method.
        self.value = None
        # New property! Keys are the inputs to this node and
        # their values are the partials of this node with
        # respect to that input.
        self.gradients = {}
        # For each inbound Node here, add this Node as an outbound Node to _that_ Node.
        # for n in self.inbound_nodes:
        #    n.outbound_nodes.append(self)
        # Sets this node as an outbound node for all of
        # this node's inputs.
        for node in inbound_nodes:
            node.outbound_nodes.append(self)

    # These will be implemented in a subclass.
    def forward(self):
        """
        Forward propagation

        Compute the output value based on 'inbound_nodes' and
        store the result in self.value
        """
        """
        Every node that uses this class as a base class will
        need to dfine its own 'forward' method.
        """
        raise NotImplementedError

    def backward(self):
        """
        Every node that uses this class as a base class will
        need to define its own 'backward' method.
        """
        raise NotImplementedError


class Input(Node):
    """
    While it may be strange to consider an input a node when
    an input is only an individual node in a node, for the sake
    of simpler code we'll still use Node as the base class.

    Think of input as collating many individual input nodes into
    a Node.
    """
    """
    A generic input into the network.
    """
    def __init__(self):
        # An Input node has no inbound nodes,
        # so no need to pass anything to the Node instantiator.
        # The base class constructor has to run to set all
        # the properties here.
        #
        # The most important property on an Input is value.
        # self.value is set during 'topological_sort' later.
        Node.__init__(self)

    # Note: Input node is the only node where the value
    # may be passed as an argument to forward().
    #
    # All other node implementations should get the value
    # of the previous node from self.inbound_nodes
    #
    # Example:
    # val0 = self.inbound_node[0].value
    def forward(self):
        # Do nothing because nothing is calculated.
        pass
        # # Overwrite the value if one is passed in.
        # if value is not None:
        #     self.value = value

    def backward(self):
        # An Input node has no inputs so the gradient (derivative)
        # is zero.
        # THe key, 'self', is reference to this object.
        self.gradients = {self: 0}
        # Weights and bias may be inputs, so you need to sum
        # the gradient from output gradients.
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1


"""
Can you augment the Add class so that it accepts
any number of nodes as input?

Hint: this may be useful:
https://docs.python.org/3/tutorial/controlflow.html#unpacking-argument-lists
"""


# class Add(Node):
#     def __init__(self, x, y):
#         Node.__init__(self, [x, y])  # calls Node's constructor
#
#     def forward(self):
#         """
#         Set the value of this node ('self.value') to the sum of its inbound_nodes.
#
#         Your code here!
#         """
#         """
#         For reference, here's the old way from the last
#         quiz. You'll want to write code here.
#         """
#         x_value = self.inbound_nodes[0].value
#         y_value = self.inbound_nodes[1].value
#         self.value = x_value + y_value


class Linear(Node):
    """
    Represents a node that performs a linear transform
    """
    def __init__(self, X, W, b):
        # Notice the ordering of the input nodes passed to the
        # Node constructor.
        # The base class (Node) constructor. Weights and bias
        # are treated like inbound nodes.
        Node.__init__(self, [X, W, b])

    # def __init__(self, inputs, weights, bias):
    #     Node.__init__(self, [inputs, weights, bias])
    #
    #     # NOTE: The weights and bias properties here are not
    #     # numbers, but rather references to other nodes.
    #     # The weight and bias values are stored within the
    #     # respective nodes.

    def forward(self):
        """
        Set self.value to the value of the linear function output.

        Your code goes here!
        """
        """
        Performs the math behind a linear transform.
        """
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value
        self.value = np.dot(X, W) + b
        # inputs = self.inbound_nodes[0].value
        # weights = self.inbound_nodes[1].value
        # bias = self.inbound_nodes[2].value
        # self.value = bias
        # for x, w in zip(inputs, weights):
        #     self.value += x * w
        # # pass

    def backward(self):
        """
        Calculates the gradient based on the output values.
        """
        # Initialize a partial for each of the inbound_nodes.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
            # Set the partial of the loss with respect to this node's inputs.
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            # Set the partial of the loss with respect to this node's weights.
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            # Set the partial of the loss with respect to this node's bias.
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)


class Sigmoid(Node):
    """
    You need to fix the '_sigmoid' and 'forward' methods.
    """
    """
    Represents a node that performs the sigmoid activation function.
    """
    def __init__(self, node):
        # The base class constructor.
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        """
        This method is separated from 'forward' because it
        will be used later with 'backward' as well.

        'x': A numpy array-like object.

        Return the result of the sigmoid function.

        Your code here!
        """
        return 1. / (1. + np.exp(-x))  # the '.' ensures that '1' is a float

    def forward(self):
        """
        Set the value of this node to the result of the
        sigmoid function, '_sigmoid'

        Your code here!
        """
        """
        Perform the sigmoid function and set the value.
        """
        # This is a dummy value to prevent numpy errors
        # if you test without changing this method.
        # self.value = -1
        input_value = self.inbound_nodes[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        """
        Calculates the gradient using the derivative of
        the sigmoid function.
        """
        # Initialize the gradients to 0.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        # sum the derivative with respect to the input over all the outputs
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
            sigmoid = self.value
            self.gradients[self.inbound_nodes[0]] += sigmoid * (1 - sigmoid) * grad_cost
            """
            TODO: Your code goes here!
            
            Set the gradients property to the gradients with respect to each input.
            
            NOTE: See the Linear node and MSE node for examples.
            """


class MSE(Node):
    def __init__(self, y, a):
        """
        The mean squared error cost function.
        Should be used as the last node for a network.
        """
        # Call the base class's constructor.
        Node.__init__(self, [y, a])

    def forward(self):
        """
        Calculates the mean squared error.
        """
        # NOTE: We reshape these to avoid possible matrix/vector broadcast
        # errors.
        #
        # For example, if we subtract an array of shape (3,) from an array of shape
        # (3,1) we get an array of shape(3,3) as the result when we want
        # an array of shape (3,1) instead.
        #
        # Making both arrays (3,1) insures the result is (3,1) and does
        # an elementwise subtraction as expected.
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)
        # TODO: your code here
        # pass

        self.m = self.inbound_nodes[0].value.shape[0]
        # Save the computed output for backward.
        self.diff = y - a
        self.value = np.mean(self.diff**2)
        # self.value = (1 / m) * np.sum(diff**2)

    def backward(self):
        """
        Calculates the gradient of the cost.

        This is the final node of the network so outbound nodes
        are not a concern.
        """
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff


"""
No need to change anything below here!
"""


def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    'feed_dict': A dictionary where the key is a 'Input' Node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


# def forward_pass(output_node, sorted_nodes):
#     """
#     Performs a forward pass through a list of sorted nodes.
#
#     Arguments:
#
#         'output_node': A node in the graph, should be the output node (have no outgoing edges).
#         'sorted_nodes': A topologically sorted list of nodes.
#
#     Returns the output Node's value
#     """
#
#     for n in sorted_nodes:
#         n.forward()
#
#     return output_node.value

# def forward_pass(graph):
#     """
#     Performs a forward pass through a list of sorted Nodes.
#
#     Arguments:
#
#         'graph': The result of calling 'topological_sort'
#     """
#     # Forward pass
#     for n in graph:
#         n.forward()

def forward_and_backward(graph):
    """
    Performs a forward pass and a backward pass through a list of sorted Nodes.

    Arguments:

        'graph': The result of calling 'topological_sort'
    """
    # Forward pass
    for n in graph:
        n.forward()

    # Backward pass
    # see: https://docs.python.org/2.3/whatsnew/section-slices.html
    for n in graph[::-1]:
        n.backward()


def sgd_update(trainables, learning_rate=1e-2):
    """
    Updates the value of each trainable with SGD.

    Arguments:

        'trainables': A list of 'Input' Nodes representing weights/biases.
        'Learning_rate': The learning rate.
    """
    # TODO: update all the 'trainables' with SGD
    # You can access and assign the value of a trainable with 'value' attribute.
    # Example:
    # for t in trainables:
    #   t.value = your implementation here
    # pass
    # Performs SGD
    #
    # Loop over the trainables
    for t in trainables:
        # Change the trainable's value by subtracting the learning rate
        # multiplied by the partial of the cost with respect to this
        # trainable.
        partial = t.gradients[t]
        t.value -= learning_rate * partial
