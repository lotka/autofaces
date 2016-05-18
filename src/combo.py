import sys
if sys.argv[1] == 'agg':
    import matplotlib
    matplotlib.use('Agg')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm
import metric

def compare(sess,data,i,autoencoder):
    input = data.train.images[i]
    output = sess.run(autoencoder['output_decoded'],{x:[input]})
    input = np.array(input).reshape(28,28)
    output = np.array(output).reshape(28,28)
    plt.figure()
    plt.subplot(211)
    plt.imshow(input,cmap='gray')
    plt.subplot(212)
    plt.imshow(output,cmap='gray')
    plt.show()

def compareall(sess,data,autoencoder,x,save=True,show=False,run_id='_'):
    plt.figure()
    for i in xrange(10):
        input = data.train.images[i]
        output = sess.run(autoencoder['output_decoded'],{x:[input]})
        input = np.array(input).reshape(28,28)
        output = np.array(output).reshape(28,28)
        plt.subplot(2,10,i+1)
        plt.imshow(input)
        plt.subplot(2,10,10+i+1)
        plt.imshow(output)

    if show:
        plt.show()
    if save:
        plt.savefig('autoencoder_graph' + str(run_id) + '.png',dpi=1000)

# Create the network
def create(x,y, layer_sizes):

    # Build the encoding layers
    next_layer_input = x
    alpha = tf.Variable(0.5,trainable=False)

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


# Get the data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def main(arg):

    alternating,pre_training_batches,combined_cost_function,iteration,batch_size,run_id = arg

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    input_size = mnist.train.images.shape[1]

    x = tf.placeholder(tf.float32, [None, input_size])
    y = tf.placeholder(tf.float32,[None,10])
    sizes = [500,400,200,50]

    autoencoder = create(x,y,sizes)
    init = tf.initialize_all_variables()
    sess.run(init)
    dual_train_step  = tf.train.GradientDescentOptimizer(0.5).minimize(autoencoder['cost_total'])
    class_train_step = tf.train.GradientDescentOptimizer(0.5).minimize(autoencoder['cost_class'])
    auto_train_step  = tf.train.GradientDescentOptimizer(0.5).minimize(autoencoder['cost_autoencoder'])

    c1_axis = np.zeros(0)
    c2_axis = np.zeros(0)
    c3_axis = np.zeros(0)
    x_axis = np.zeros(0)

    if pre_training_batches > 0:
        """
        PRETRAIN
        """
        print 'pre-train autoencoder:'
        for i in tqdm(range(pre_training_batches)):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(auto_train_step, feed_dict={x: batch_xs, y: batch_ys})


    # do 1000 training stepscomp
    # print 'i\ttot\tclass\tauto'
    for i in tqdm(range(iteration)):
        # Train classifier
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        if combined_cost_function:
            sess.run(dual_train_step, feed_dict={x: batch_xs, y: batch_ys})
        else:
            sess.run(class_train_step, feed_dict={x: batch_xs, y: batch_ys})


        if alternating:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(auto_train_step, feed_dict={x: batch_xs, y: batch_ys})

        if i % 100 == 0:
            batch_xs, batch_ys = mnist.validation.next_batch(batch_size)
            c1 = sess.run(autoencoder['cost_total'], feed_dict={x: batch_xs, y: batch_ys})
            c2 = sess.run(autoencoder['cost_class'], feed_dict={x: batch_xs, y: batch_ys})
            c3 = sess.run(autoencoder['cost_autoencoder'], feed_dict={x: batch_xs, y: batch_ys})
            # print 'wtf', c
            x_axis = np.append(x_axis,i)
            c1_axis = np.append(c1_axis,c1)
            c2_axis = np.append(c2_axis,c2)
            c3_axis = np.append(c3_axis,c3)

            # print i,
            # print c,
            # print sess.run(autoencoder['cost_class'], feed_dict={x: batch_xs, y: batch_ys}),
            # print sess.run(autoencoder['cost_autoencoder'], feed_dict={x: batch_xs, y: batch_ys})

            # print i, " original", batch[0]
            # print i, " decoded", sess.run(autoencoder['decoded'], feed_dict={x: batch})

    # compare(sess,mnist,2)
    compareall(sess,mnist,autoencoder,x,save=True,show=False,run_id=run_id)
    fig = plt.figure()
    plt.plot(x_axis,c1_axis,label='cost_total')
    plt.plot(x_axis,c2_axis,label='cost_class')
    plt.plot(x_axis,c3_axis,label='cost_autoencoder')
    # plt.show()
    plt.legend()
    plt.savefig('graph' + str(run_id))

    metric.metric(autoencoder,sess,y,mnist,x,'log'+str(run_id)+'.txt')
    sess.close()

iteration = 100000
pre_train_size = 5000
args = []
args.append((True,  0,False,iteration,100,0))
args.append((True,  0,True,iteration,100,1))
args.append((True,  pre_train_size,False,iteration,100,2))
args.append((True,  pre_train_size,True,iteration,100,3))
args.append((False, 0,False,iteration,100,4))
args.append((False, 0,True,iteration,100,5))
args.append((False, pre_train_size,False,iteration,100,6))
args.append((False, pre_train_size,True,iteration,100,7))



for a in args:
    main(a)
