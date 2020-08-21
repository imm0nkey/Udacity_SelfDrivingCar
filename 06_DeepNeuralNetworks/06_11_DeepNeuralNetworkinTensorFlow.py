# TensorFlow MNIST
from tensorflow.examples.tutorials.mnist import input_data
minst = input_data.read_data_sets(".", one_hot=True, reshape=False)

# Learning Parameters
import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 128  # Decrease batch size if you don't have enough memory
display_step = 1

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Hidden Layer Parameters
n_hidden_layer = 256  # layer number of features

# Weights and Biases
# Store layers weight & bias
weights = {
    'hidden_layer': tf.Variable(tf.random_normal())
    'out': tf.Variable(tf.random)
}
biases = {


}

# Input
# tf Graph input
x =
y =

x_flat =

# Multilayer Preceptron

