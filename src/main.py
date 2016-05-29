import numpy as np
import h5py
import metric
import os
from scipy import ndimage
import scipy
import socket
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyexp.pyexp import PyExp
import math
import tensorflow as tf
import sys

def prefix(i,zeros):
    s = str(i)
    while(len(s) < zeros):
        s = '0' + s
    return s

def main(data,experiment):

    def weight_variable(shape):
      # initial = tf.truncated_normal(shape, stddev=0.1)
      # return tf.Variable(initial)
      return tf.Variable(tf.random_uniform(shape, -1.0 / math.sqrt(shape[0]), 1.0 / math.sqrt(shape[0])))

    def bias_variable(shape):
      # initial = tf.constant(0.1, shape=shape)
      # return tf.Variable(initial)
      return tf.Variable(tf.constant(0.1,shape=shape))

    def conv2d(x, W,padding='SAME'):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

    def max_pool_axb(x,a,b):
        return tf.nn.max_pool(x, ksize=[1, a, b, 1],strides=[1, a, b, 1], padding='SAME')
    size = experiment['image_size']
    x = tf.placeholder(np.float32, shape=[None, size,size])
    x_ = tf.reshape(x, [-1,size,size,1])
    y_ = tf.placeholder(np.float32, shape=[None, 12])

    print 'Input: ', x_.get_shape()
    print 'Outut: ', y_.get_shape()
    convolution_network = False
    if experiment['network'] == 'convolution_gudi_2015':
        W_conv1 = weight_variable([5, 5, 1, 64])
        b_conv1 = bias_variable([64])

        print 'Convolution 1 Shape: ', W_conv1.get_shape(), ' with bias ', b_conv1.get_shape(), ' padding VALID'

        h_conv1 = tf.nn.relu(conv2d(x_, W_conv1,padding='VALID') + b_conv1)
        print ' output : ', h_conv1.get_shape()
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1],padding='SAME')
        print 'Max pool 3x3, stride 2x2, padding SAME'
        print ' output : ', h_pool1.get_shape()

        W_conv2 = weight_variable([5, 5, 64, 64])
        b_conv2 = bias_variable([64])

        print 'Convolution 2 Shape: ', W_conv2.get_shape(), ' with bias ', b_conv2.get_shape(), ' padding VALID'

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2,padding="VALID") + b_conv2)
        print ' output : ', h_conv2.get_shape()

        W_conv3 = weight_variable([4, 4, 64, 128])
        b_conv3 = bias_variable([128])

        print 'Convolution 3 Shape: ', W_conv3.get_shape(), ' with bias ', b_conv3.get_shape(), ' padding VALID'

        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3,padding="VALID") + b_conv3)
        print ' output : ', h_conv3.get_shape()
        s = 128*15*15
        print ' flatten to ', s
        flat_1 = tf.reshape(x_, [-1,s])
    if experiment['network'] == 'fullyconnected_subnetwork_gudi_2015':
        s = 48*48
        print ' flatten to ', s
        flat_1 = tf.reshape(x_, [-1,s])
    # print h_conv3_flat.get_shape()


    W1 = weight_variable([s,3072])
    b1 = bias_variable([3072])
    print 'Fully connected matrix 1 : ', W1.get_shape(), 'with bias',b1.get_shape()

    full_1 = tf.nn.relu(tf.matmul(flat_1,W1)+b1)

    W2 = weight_variable([3072,12])
    b2 = bias_variable([12])

    print 'Fully connected matrix 2 : ', W2.get_shape(), 'with bias',b2.get_shape()

    tmp = tf.nn.softmax(tf.matmul(full_1,W2) + b2)
    y_conv = tf.transpose(tf.mul(tf.reduce_sum(tf.cast(y_,tf.float32),1),tf.transpose(tmp)))
    # y_conv = tmp
    #*tf.reduce_sum(y_,1))

    print 'Output shape : ', y_conv.get_shape()


    THRESHOLD = experiment['threshold']
    a = tf.cast(tf.greater(y_conv,THRESHOLD),tf.float32)
    b = tf.cast(tf.greater(y_,THRESHOLD),tf.float32)
    c = tf.abs(a-b)
    ab_sum = tf.reduce_sum(a,1)+tf.reduce_sum(b,1)
    accuracy = tf.reduce_mean(tf.Variable(1.0)-tf.reduce_sum(c,1)/ab_sum)

    cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
    lmsq_loss = tf.sqrt(tf.reduce_mean(tf.square(y_conv-y_)))

    if experiment['cost_function'] == 'cross_entropy':
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    else:
        train_step = tf.train.GradientDescentOptimizer(0.10).minimize(lmsq_loss)

    if socket.gethostname() == 'ux305':
        sess = tf.Session()
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)

    init = tf.initialize_all_variables()
    sess.run(init)

    def roc_area(y_true, y_score,feed_dict):
        for i in xrange(y_true.shape[1]):
            trues = y_true[:,i]
            scores = sess.run(y_score,feed_dict)[:,i]
            if trues.sum() != 0.0:
                # print trues,scores
                fpr, tpr, t = roc_curve(trues,scores)
                print fpr, tpr, t
                roc_auc = auc(fpr, tpr)
                print roc_auc
            else:
                print '?',
        print

    N = experiment['iterations']
    random_batch = experiment['batch_randomisation']
    batch_size = experiment['batch_size']
    x_axis = np.zeros(N)
    lmsq_axis = np.zeros((2,N))
    cent_axis = np.zeros((2,N))
    accu_axis = np.zeros((2,N))
    train_auac_axis = np.zeros((N,12,4))
    test_auac_axis = np.zeros((N,12,4))
    train_confusion = []
    test_confusion = []
    print 'i    test    train    test    train    test    train'
    print 'i    lmsq    lmsq    cent    cent    accu    accu'
    ref = sess.run(tf.reduce_sum(data.train.next_batch(-1,False)[1],0))
    # raw_input('Fuck off')
    path = experiment.get_path()
    for i in xrange(N):
        print i
        batch_x, batch_y = data.train.next_batch(batch_size,random_batch)
        t_batch_x, t_batch_y = data.test.next_batch(batch_size,random_batch)
        train = {x: batch_x, y_: batch_y}
        test = {x: t_batch_x, y_: t_batch_y}
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
        x_axis[i] = i
        e = [lmsq_loss,cross_entropy,accuracy]
        lmsq_axis[0,i],cent_axis[0,i],accu_axis[0,i] = sess.run(e, feed_dict={x: t_batch_x, y_: t_batch_y})
        lmsq_axis[1,i],cent_axis[1,i],accu_axis[1,i] = sess.run(e, feed_dict={x: batch_x, y_: batch_y})
        # print i,'   \t',
        # print round(lmsq_axis[0,i],2),' ',
        # print round(lmsq_axis[1,i],2),' ',
        # print round(cent_axis[0,i],2),' ',
        # print round(cent_axis[1,i],2),' ',
        # print round(accu_axis[0,i],2),' ',
        # print round(accu_axis[1,i],2),' '
        y_out = sess.run(y_conv, feed_dict={x: batch_x, y_: batch_y})
        # print y_out
        # print batch_y
        # print ref
        train_res, train_conf = metric.multi_eval(y_out,batch_y)
        train_auac_axis[i,:,:] = train_res
        test_res, test_conf = metric.multi_eval(y_out,t_batch_y)
        test_auac_axis[i,:,:] = test_res

        train_confusion.append(train_conf)
        test_confusion.append(test_conf)

        # print res,
        # print
    sess.close()

    ssv_path = os.path.join(path,'numerical_data')
    if not os.path.isdir(ssv_path):
        os.mkdir(ssv_path)
    np.savetxt(os.path.join(ssv_path,'x_axis.ssv'),x_axis)
    np.savetxt(os.path.join(ssv_path,'lmsq.ssv'),lmsq_axis)
    np.savetxt(os.path.join(ssv_path,'cross_entropy.ssv'),cent_axis)
    np.savetxt(os.path.join(ssv_path,'naive_accuracy.ssv'),accu_axis)
    np.savez(os.path.join(ssv_path,'per_au_accuracy.npz'),
             train_metrics=train_auac_axis,
             test_metrics=test_auac_axis,
             train_confusion=train_confusion,
             test_confusion=test_confusion
             )


experiment =  PyExp(config_file='config/cnn.yaml',path='data')
if experiment['data']['dataset'] == 'disfa':
    import disfa
    data = disfa.Disfa(number_of_subjects=3,train_prop=1,valid_prop=1,test_prop=1)
    main(data,experiment)
else:
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    main(mnist,experiment)
