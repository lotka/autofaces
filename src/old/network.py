import tensorflow as tf
import math
import numpy as np

# Create the network
def create(x,y, layer_sizes,_alpha=0.5):

    # Build the encoding layers
    next_layer_input = x
    alpha = tf.Variable(_alpha,trainable=False)

    encoding_matrices = []
    for dim in layer_sizes:
        print next_layer_input.get_shape()
        input_dim = int(next_layer_input.get_shape()[1])

        # Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
        W = tf.Variable(tf.random_uniform([input_dim, dim], -1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(input_dim)))

        # Initialize b to zero
        b = tf.Variable(tf.zeros([dim]))

        # We are going to use tied-weights so store the W matrix for later reference.
        encoding_matrices.append(W)

        output = tf.nn.relu(tf.matmul(next_layer_input,W) + b)

        # the input into the next layer is the output of this layer
        next_layer_input = output

    # The fully encoded x value is now stored in the next_layer_input
    encoded_x = next_layer_input

    input_dim = int(encoded_x.get_shape()[1])
    output_dim = int(y.get_shape()[1])
    W = tf.Variable(tf.random_uniform([input_dim,output_dim],-1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(input_dim)))
    b = tf.Variable(tf.zeros([output_dim]))
    class_layer = tf.nn.relu(tf.matmul(encoded_x,W) + b)

    # build the reconstruction layers by reversing the reductions
    layer_sizes.reverse()
    encoding_matrices.reverse()


    for i, dim in enumerate(layer_sizes[1:] + [ int(x.get_shape()[1])]) :
        # we are using tied weights, so just lookup the encoding matrix for this step and transpose it
        W = tf.transpose(encoding_matrices[i])
        b = tf.Variable(tf.zeros([dim]))
        output = tf.nn.relu(tf.matmul(next_layer_input,W) + b)
        next_layer_input = output

    # the fully encoded and reconstructed value of x is here:
    reconstructed_x = next_layer_input

    # Cost function for the auto encoder
    auto_cost = tf.sqrt(tf.reduce_mean(tf.square(x-reconstructed_x)))

    # Cost function for the classifier
    class_cost = tf.sqrt(tf.reduce_mean(tf.square(y-class_layer)))

    return {
        'output_class' :  class_layer,
        'output_encoded': encoded_x,
        'output_decoded': reconstructed_x,
        'cost_total' : auto_cost*(1-alpha)+class_cost*alpha,
        'cost_class' : class_cost,
        'cost_autoencoder' : auto_cost,
        'alpha' : alpha
    }
