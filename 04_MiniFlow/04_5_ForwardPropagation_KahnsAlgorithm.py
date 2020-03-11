"""
This script builds and runs a graph with miniflow.

There is no need to change anything to solve this quiz!

However, feel free to play with the network! Can you also
build a network that solves the equation below?

(x + y) + y
"""
"""
Test your network here!

No need to change this code, but feel free to tweak it
to test your network!

Make your changes to backward method of the Sigmoid class in miniflow.py
"""

import numpy as np
from miniflow import *

X, W, b = Input(), Input(), Input()
y = Input()
f = Linear(X, W, b)
a = Sigmoid(f)
cost = MSE(y, a)
# y, a = Input(), Input()
# cost = MSE(y, a)
# X, W, b = Input(), Input(), Input()
# inputs, weights, bias = Input(), Input(), Input()
# x, y, z = Input(), Input(), Input()
# x, y = Input(), Input()

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2.], [3.]])
b_ = np.array([-3.])
y_ = np.array([1, 2])

feed_dict = {
    X: X_,
    y: y_,
    W: W_,
    b: b_,
}
# y_ = np.array([1, 2, 3])
# a_ = np.array([4.5, 5, 10])
#
# feed_dict = {y: y_, a: a_}
# f = Linear(X, W, b)
# g = Sigmoid(f)
# f = Linear(inputs, weights, bias)
# f = Add(x, y, z)
# f = Add(x, y)

# X_ = np.array([[-1., -2.], [-1, -2]])
# W_ = np.array([[2., -3], [2., -3]])
# b_ = np.array([-3., -5])

# feed_dict = {X: X_, W: W_, b: b_}
# feed_dict = {
#     inputs: [6, 14, 3],
#     weights: [0.5, 0.25, 1.4],
#     bias: 2
# }
# feed_dict = {x: 4, y: 5, z: 10}
# feed_dict = {x: 10, y: 5}

graph = topological_sort(feed_dict)
forward_and_backward(graph)
# return the gradients for each Input
gradients = [t.gradients[t] for t in [X, y, W, b]]
# forward pass
# forward_pass(graph)
# output = forward_pass(g, graph)
# sorted_nodes = topological_sort(feed_dict)
# output = forward_pass(f, sorted_nodes)

# NOTE: because topological_sort set the values for the 'Input' nodes we could also access
# the value for x with x.value (same goes for y).
# should output 19 for 3 variables
# print("{} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], output))
# should be 12.7 with this example
"""
Output should be:
[[-9., 4.],
[-9., 4.]]
"""
"""
Output should be:
[[  1.23394576e-04   9.82013790e-01]
 [  1.23394576e-04   9.82013790e-01]]
"""
# print(cost.value)
# print(output)

"""
Expected output

[array([[ -3.34017280e-05,  -5.01025919e-05],
       [ -6.68040138e-05,  -1.00206021e-04]]), array([[ 0.9999833],
       [ 1.9999833]]), array([[ 5.01028709e-05],
       [ 1.00205742e-04]]), array([ -5.01028709e-05])]
"""
print(gradients)
