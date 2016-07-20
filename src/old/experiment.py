import tqdm

from network import create,tf,np
from src.pyexp import PyExp

debug = True

if debug:
    import shutil
    import os
    if os.path.isdir('test_data'):
        shutil.rmtree('test_data')

experiment =  PyExp(config_file='config/test.yaml',path='test_data')

# Get the data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z

def network_01(config):

    batch_size = config['batch_size']
    sess = tf.Session()
    input_size = mnist.train.images.shape[1]

    x = tf.placeholder(tf.float32, [None, input_size])
    y = tf.placeholder(tf.float32,[None,10])

    autoencoder = create(x,y,config['autoencoder'])
    init = tf.initialize_all_variables()
    sess.run(init)
    dual_train_step  = tf.train.GradientDescentOptimizer(0.5).minimize(autoencoder['cost_total'])
    class_train_step = tf.train.GradientDescentOptimizer(0.5).minimize(autoencoder['cost_class'])
    auto_train_step  = tf.train.GradientDescentOptimizer(0.5).minimize(autoencoder['cost_autoencoder'])

    c1_axis = np.zeros(0)
    c2_axis = np.zeros(0)
    c3_axis = np.zeros(0)
    x_axis = np.zeros(0)

    if config['pre_training_batches'] > 0:
        """
        PRETRAIN
        """
        print 'pre-train autoencoder:'
        for i in tqdm(range(pre_training_batches)):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(auto_train_step, feed_dict={x: batch_xs, y: batch_ys})


    # do 1000 training stepscomp
    # print 'i\ttot\tclass\tauto'
    N = config['iterations']
    print N
    for i in range(N):
        # Train classifier
        batch_xs, batch_ys = mnist.train.next_batch(config['batch_size'])

        if config['combined_cost_function']:
            sess.run(dual_train_step, feed_dict={x: batch_xs, y: batch_ys})
        else:
            sess.run(class_train_step, feed_dict={x: batch_xs, y: batch_ys})


        if config['alternating']:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(auto_train_step, feed_dict={x: batch_xs, y: batch_ys})

        if i % 100 == 0:
            batch_xs, batch_ys = mnist.validation.next_batch(batch_size)
            c1 = sess.run(autoencoder['cost_total'], feed_dict={x: batch_xs, y: batch_ys})
            c2 = sess.run(autoencoder['cost_class'], feed_dict={x: batch_xs, y: batch_ys})
            c3 = sess.run(autoencoder['cost_autoencoder'], feed_dict={x: batch_xs, y: batch_ys})
            print i,c1,c2,c3
            # x_axis = np.append(x_axis,i)
            # c1_axis = np.append(c1_axis,c1)
            # c2_axis = np.append(c2_axis,c2)
            # c3_axis = np.append(c3_axis,c3)
    sess.close()


runs = experiment['runs']

for i, conf in runs.iteritems():
    print i,conf
    network_01(merge_two_dicts(conf,experiment['global']))
