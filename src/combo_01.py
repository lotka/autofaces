import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import random

def create(x,y, layer_sizes):

    # Build the encoding layers
    next_layer_input = x
    alpha = tf.Variable(1.0,trainable=False)

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

    # alpha = 0.5
    return {
        'encoded': encoded_x,
        'decoded': reconstructed_x,
        'cost' : tf.sqrt(tf.reduce_mean(tf.square(x-reconstructed_x)))*(1-alpha)+tf.sqrt(tf.reduce_mean(tf.square(y-encoded_x)))*alpha,
        'alpha' : alpha
    }

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.Session()
input_size = mnist.train.images.shape[1]
x = tf.placeholder(tf.float32, [None, input_size])
y = tf.placeholder(tf.float32,[None,10])
sizes = [500,400,300,10]
autoencoder = create(x,y,sizes)
init = tf.initialize_all_variables()
sess.run(init)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(autoencoder['cost'])

x_axis = np.zeros(0)
y_axis = np.zeros(0)
# do 1000 training steps
for i in range(100):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
    if i % 100 == 0:
        c = sess.run(autoencoder['cost'], feed_dict={x: batch_xs, y: batch_ys})
        print i, " cost", c, sess.run(autoencoder['alpha'])
        x_axis = np.append(x_axis,i)
        y_axis = np.append(y_axis,c)
        # print i, " original", batch[0]
        # print i, " decoded", sess.run(autoencoder['decoded'], feed_dict={x: batch})


def compare(sess,data,i):
    input = data.train.images[i]
    output = sess.run(autoencoder['decoded'],{x:[input]})
    input = np.array(input).reshape(28,28)
    output = np.array(output).reshape(28,28)
    fig = plt.figure()
    plt.subplot(211)
    plt.imshow(input,cmap='gray')
    plt.subplot(212)
    plt.imshow(output,cmap='gray')
    plt.show()

def compareall(sess,data):
    plt.figure()
    for i in xrange(10):
        input = data.train.images[i]
        output = sess.run(autoencoder['decoded'],{x:[input]})
        input = np.array(input).reshape(28,28)
        output = np.array(output).reshape(28,28)
        plt.subplot(2,10,i+1)
        plt.imshow(input)
        plt.subplot(2,10,10+i+1)
        plt.imshow(output)
    plt.show()

compare(sess,mnist,2)
compareall(sess,mnist)
plt.figure()
plt.plot(x_axis,y_axis)
plt.show()
