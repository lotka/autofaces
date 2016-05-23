""" Deep Auto-Encoder implementation

    An auto-encoder works as follows:

    Data of dimension k is reduced to a lower dimension j using a matrix multiplication:
    softmax(W*x + b)  = x'

    where W is matrix from R^k --> R^j

    A reconstruction matrix W' maps back from R^j --> R^k

    so our reconstruction function is softmax'(W' * x' + b')

    Now the point of the auto-encoder is to create a reduction matrix (values for W, b)
    that is "good" at reconstructing  the original data.

    Thus we want to minimize  ||softmax'(W' * (softmax(W *x+ b)) + b')  - x||

    A deep auto-encoder is nothing more than stacking successive layers of these reductions.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import random

def create(x,rho_hat,layer_sizes):

    # Build the encoding layers
    next_layer_input = x

    rho = []
    cost_kl = tf.Variable(0.0)
    j = 0
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
        rho.append(tf.reduce_mean(output))
        cost_kl += tf.abs( rho[-1]-rho_hat[j] )
        j += 1
        print 'AOSUHFOIASHFOIA', len(rho)

        # the input into the next layer is the output of this layer
        next_layer_input = output

    # The fully encoded x value is now stored in the next_layer_input
    encoded_x = next_layer_input

    # build the reconstruction layers by reversing the reductions
    layer_sizes.reverse()
    encoding_matrices.reverse()


    for i, dim in enumerate(layer_sizes[1:] + [ int(x.get_shape()[1])]) :
        # we are using tied weights, so just lookup the encoding matrix for this step and transpose it
        print next_layer_input.get_shape()
        W = tf.transpose(encoding_matrices[i])
        b = tf.Variable(tf.zeros([dim]))
        output = tf.nn.relu(tf.matmul(next_layer_input,W) + b)
        next_layer_input = output
        rho.append(tf.reduce_mean(output))
        cost_kl += tf.abs( rho[-1]-rho_hat[j] )
        j += 1
        print 'AOSUHFOIASHFOIA', len(rho)

    # the fully encoded and reconstructed value of x is here:
    reconstructed_x = next_layer_input

    return {
        'rho' : rho,
        'cost_kl': cost_kl,
        'encoded': encoded_x,
        'decoded': reconstructed_x,
        'cost' : tf.sqrt(tf.reduce_mean(tf.square(x-reconstructed_x)))+cost_kl
    }

def simple_test():
    sess = tf.Session()
    x = tf.placeholder("float", [None, 4])
    autoencoder = create(x, [2])
    init = tf.initialize_all_variables()
    sess.run(init)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(autoencoder['cost'])


    # Our dataset consists of two centers with gaussian noise w/ sigma = 0.1
    c1 = np.array([0,0,0.5,0])
    c2 = np.array([0.5,0,0,0])

    # do 1000 training steps
    for i in range(2000):
        # make a batch of 100:
        batch = []
        for j in range(100):
            # pick a random centroid
            if (random.random() > 0.5):
                vec = c1
            else:
                vec = c2
            batch.append(np.random.normal(vec, 0.1))
        sess.run(train_step, feed_dict={x: np.array(batch)})
        if i % 100 == 0:
            print i, " cost", sess.run(autoencoder['cost'], feed_dict={x: batch})


def deep_test():
    sess = tf.Session()
    start_dim = 200
    x = tf.placeholder("float", [None, start_dim])
    autoencoder = create(x, [4, 2])
    init = tf.initialize_all_variables()
    sess.run(init)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(autoencoder['cost'])


    # Our dataset consists of two centers with gaussian noise w/ sigma = 0.1
    c1 = np.zeros(start_dim)
    c1[0] = 1

    print c1

    c2 = np.zeros(start_dim)
    c2[1] = 1

    x_axis = np.zeros(0)
    y_axis = np.zeros(0)

    # do 1000 training steps
    for i in range(5000):
        # make a batch of 100:
        batch = []
        for j in range(1):
            # pick a random centroid
            if (random.random() > 0.5):
                vec = c1
            else:
                vec = c2
            batch.append(np.random.normal(vec, 0.1))
        sess.run(train_step, feed_dict={x: np.array(batch)})
        if i % 100 == 0:
            c = sess.run(autoencoder['cost'], feed_dict={x: batch})
            print i, " cost", c
            x_axis = np.append(x_axis,i)
            y_axis = np.append(y_axis,c)
            # print i, " original", batch[0]
            # print i, " decoded", sess.run(autoencoder['decoded'], feed_dict={x: batch})
    plt.figure()
    plt.plot(x_axis,y_axis)
    plt.show()


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.Session()
input_size = mnist.train.images.shape[1]
x = tf.placeholder(tf.float32, [None, input_size])
sizes = [400,10]
rho_hat = tf.placeholder(tf.float32, [(len(sizes))*2])
autoencoder = create(x,rho_hat,sizes)
init = tf.initialize_all_variables()
sess.run(init)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(autoencoder['cost'])

x_axis = np.zeros(0)
y_axis = np.zeros(0)
# do 1000 training steps
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    r = sess.run(autoencoder['rho'],feed_dict={x: batch_xs})
    sess.run(train_step, feed_dict={x: batch_xs, rho_hat: r})
    if i % 100 == 0:
        c = sess.run(autoencoder['cost'], feed_dict={x: batch_xs, rho_hat: r})
        print i, " cost", c,
        print sess.run(autoencoder['cost_kl'], feed_dict={x: batch_xs, rho_hat: r})
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

exit()
compare(sess,mnist,2)
plt.figure()
plt.plot(x_axis,y_axis)
plt.show()


"""
TODO:
Write the prototype in mnist, getting all of the validation techniques done so that we can go to DISFA
Always save results, store it in a good format
Use:
* train
* validation
* test

Try all the combinations
* classification one
* autoencoder
* ....
Weighting of the cost function
"""
