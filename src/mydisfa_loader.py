"""
Lets try and load the DISFA data
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from scipy import ndimage
import scipy

def prefix(i,zeros):
    s = str(i)
    while(len(s) < zeros):
        s = '0' + s
    return s

def get_face(i,subject,warped=True,path='/home/luka/v/hmi/projects/sebastian/DISFA',resize=None):
    if warped:
        s = 'SN' + prefix(subject,3)
        s = os.path.join(s,s + '_warp_frame' + prefix(i,4) + '.png')
        s = os.path.join('Videos_warped',s)
        data = ndimage.imread(os.path.join(path,s))
        if resize != None:
            data = scipy.misc.imresize(data,resize)
        return data

def get_subject_names(path='/home/luka/v/hmi/projects/sebastian/DISFA'):
    subjects = []
    for i in xrange(1000):
        s = os.path.join(path,'Videos/SN'+prefix(i,3))
        if os.path.isdir(s):
            subjects.append(i)
    return subjects

def get_au_names(path='/home/luka/v/hmi/projects/sebastian/DISFA'):
    aus = []
    for i in xrange(1000):
        s = os.path.join(path,'Labels/AU'+str(i))
        if os.path.isdir(s):
            aus.append(i)
    return aus

# %matplotlib inline
path = '/vol/hmi/projects/sebastian/DISFA'

subjects = get_subject_names(path=path)
aus = get_au_names(path=path)
print aus
print subjects

print path
data = get_face(3000,8,resize=[48,48],path=path)
print data.shape
plt.figure()
plt.imshow(data,cmap='gray')
plt.show()

def get_au(au,subject,path='/home/luka/v/hmi/projects/sebastian/DISFA'):
    s = os.path.join('Labels','AU'+str(au))
    name = 'SN'+prefix(subject,3)+'_labels_AU'+str(au)+'.mat'
    s = os.path.join(s,name)
    s = os.path.join(path,s)
    f = h5py.File(s,'r')
    return f['labels'][()][0]

au_combined = []
for i in xrange(26):
    if i in aus:
        au_combined.append(get_au(au=i,subject=4,path=path))
    else:
        au_combined.append(np.zeros(4845))

subject_data = []
for i in range(1,4846):
    print i,
    subject_data.append(get_face(i,8,resize=[48,48],path=path))

Labels = np.array(au_combined,dtype=float).transpose()
Images = np.array(subject_data)

for i in xrange(Labels.shape[0]):
    for j in xrange(Labels.shape[1]):
        if Labels[i,j] > 0:
            Labels[i,j] = 1
        else:
            Labels[i,j] = 0

Images = Images.astype(float)/float(Images.max())

import math
import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W,padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

def max_pool_axb(x,a,b):
    return tf.nn.max_pool(x, ksize=[1, a, b, 1],strides=[1, a, b, 1], padding='SAME')

x = tf.placeholder(np.float32, shape=[None, 48,48])
x_ = tf.reshape(x, [-1,48,48,1])
y_ = tf.placeholder(np.float32, shape=[None, 26])

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

# W_conv3 = weight_variable([4, 4, 64, 128])
# b_conv3 = bias_variable([128])

# h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3,padding="VALID") + b_conv3)

# print h_conv3.get_shape()
# print h_pool2.get_shape()

h_conv3_flat = tf.reshape(h_conv2, [-1, 207360])
print h_conv3_flat.get_shape()

W1 = tf.Variable(tf.random_uniform([207360,500], -1.0 / math.sqrt(288000), 1.0 / math.sqrt(288000)))
b1 = tf.Variable(tf.zeros([500]))

full_1 = tf.nn.relu(tf.matmul(h_conv3_flat,W1)+b1)
print full_1.get_shape()

W2 = tf.Variable(tf.random_uniform([500,26], -1.0 / math.sqrt(3072), 1.0 / math.sqrt(3072)))
b2 = tf.Variable(tf.zeros([26]))

y_conv = tf.nn.softmax(tf.matmul(full_1,W2) + b2)
print y_conv.get_shape()

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()

init = tf.initialize_all_variables()
sess.run(init)

print Images.shape
print Labels.shape
# print type(Images[0,0,0])
print sess.run(train_step, feed_dict={x: Images[:10,:,:], y_: Labels[:10,:]})
# print sess.run(loss, feed_dict={x: Images, y: Labels})

sess.close()
