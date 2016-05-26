"""
Lets try and load the DISFA data
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from scipy import ndimage
import scipy
import seb.data_array as data_array
import seb.disfa as disfa

def prefix(i,zeros):
    s = str(i)
    while(len(s) < zeros):
        s = '0' + s
    return s

import math
import tensorflow as tf

all_images = np.empty((0, 48, 48))
#for s in [1]:
for s in [1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,21,23,24,25,26,27,28,29,30,31,32]:
    images = disfa.disfa['images'][s][:]
    resized_images = np.zeros((images.shape[0],48,48))
    for i in xrange(images.shape[0]):
        resized_images[i,:,:] = scipy.misc.imresize(images[i,:,:],[48,48]).astype(float)/float(255)
    del images
    all_images = np.append(all_images,resized_images,axis=0)
    print all_images.shape

targets_all, id_array = data_array.IndicesCollection(disfa.disfa_ic_all).getitem(disfa.disfa['AUall'])

targets_all = targets_all.astype(float)
x,y = targets_all.shape
for i in xrange(x):
    for j in xrange(y):
        if targets_all[i,j] > 0:
            targets_all[i,j] = 1.0
        else:
            targets_all[i,j] = 0.0
    sum = float(targets_all[i,:].sum())
    if sum != 0.0:
        targets_all[i,:] = targets_all[i,:]/sum

def weight_variable(shape):
  # initial = tf.truncated_normal(shape, stddev=0.1)
  # return tf.Variable(initial)
  return tf.Variable(tf.random_uniform(shape, -1.0 / math.sqrt(shape[0]), 1.0 / math.sqrt(shape[0])))

def bias_variable(shape):
  # initial = tf.constant(0.1, shape=shape)
  # return tf.Variable(initial)
  return tf.Variable(tf.constant(0.5,shape=shape))

def conv2d(x, W,padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

def max_pool_axb(x,a,b):
    return tf.nn.max_pool(x, ksize=[1, a, b, 1],strides=[1, a, b, 1], padding='SAME')

x = tf.placeholder(np.float32, shape=[None, 48,48])
x_ = tf.reshape(x, [-1,48,48,1])
y_ = tf.placeholder(np.float32, shape=[None, 12])

W_conv1 = weight_variable([5, 5, 1, 64])
b_conv1 = bias_variable([64])

h_conv1 = tf.nn.relu(conv2d(x_, W_conv1,padding='VALID') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1],padding='SAME')

print h_conv1.get_shape()
print h_pool1.get_shape()

W_conv2 = weight_variable([5, 5, 64, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2,padding="VALID") + b_conv2)

print h_conv2.get_shape()

W_conv3 = weight_variable([4, 4, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3,padding="VALID") + b_conv3)

print h_conv3.get_shape()

s = 128*15**2
flat_1 = tf.reshape(h_conv3, [-1,s])
# print h_conv3_flat.get_shape()

W1 = weight_variable([s,500])
b1 = bias_variable([500])

full_1 = tf.nn.relu(tf.matmul(flat_1,W1)+b1)
print full_1.get_shape()

W2 = weight_variable([500,12])
b2 = bias_variable([12])

y_conv = tf.nn.sigmoid(tf.matmul(full_1,W2) + b2)
THRESHOLD = 0.3
a = tf.cast(tf.greater(y_conv,THRESHOLD),tf.float32)
b = tf.cast(tf.greater(y_,THRESHOLD),tf.float32)
c = tf.abs(a-b)
ab_sum = tf.reduce_sum(a,1)+tf.reduce_sum(b,1)
accuracy = tf.reduce_mean(tf.Variable(1.0)-tf.reduce_sum(c,1)/ab_sum)
print y_conv.get_shape()
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

lmsq_loss = tf.sqrt(tf.reduce_mean(tf.square(y_conv-y_)))
#train_step = tf.train.GradientDescentOptimizer(0.10).minimize(lmsq_loss)
#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

init = tf.initialize_all_variables()
sess.run(init)
import decimal
print all_images.shape
print targets_all.shape
# print type(Images[0,0,0])
o = 1000
i_validation,i_train = np.array_split(all_images, 2,axis=0)
l_validation,l_train = np.array_split(targets_all,2,axis=0)

c = int(i_train.shape[0]/1000.0 - 1)
#c = 100000
for i in xrange(0,c):
    L = i*o
    R = i*o + o
    batch_x = i_train[L:R,:,:]
    batch_y = l_train[L:R,:]
    vbatch_x = i_validation[L:R,:,:]
    vbatch_y = l_validation[L:R,:]
    bs = batch_y.shape
    vs = vbatch_y.shape
    print bs,vs
    print L,R,
    'train step', sess.run(train_step, feed_dict={x: batch_x, y_: batch_y}),
    print 'lmsq_loss', sess.run(lmsq_loss, feed_dict={x: vbatch_x, y_: vbatch_y}),
    print 'centropy', sess.run(cross_entropy, feed_dict={x: vbatch_x, y_: vbatch_y}),
    # print 'correct_prediction', sess.run(correct_prediction, feed_dict={x: batch_x, y_: batch_y})
    print 'accuracy', sess.run(accuracy, feed_dict={x: vbatch_x, y_: vbatch_y}),
    out =  sess.run(y_conv, feed_dict={x: batch_x, y_: batch_y})
    if(i < 10 or i > 1500):
        np.savetxt('out_'+str(i)+'.txt',out)
        np.savetxt('gut_'+str(i)+'.txt',batch_y)
    print 'output range: ', out.min(),out.max(),
    print

sess.close()
