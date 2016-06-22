#!/usr/bin/python
import etc
import tensorflow as tf
reload(tf)
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyexp.pyexp import PyExp
import math
import sys
import math
from os.path import join
from tqdm import tqdm
from model import cnn,expand_labels,batched_feed
import yaml
import disfa
path = sys.argv[1]
# Load yaml config file
config = PyExp(config_file=join(path,'config.yaml'),make_new=False)
# Load data
data = disfa.Disfa(config['data'])
config.update('data',data.config,save=False)

# Load model
x,y_,train_step,loss, y_conv,output_dim,keep_prob,lmsq_loss,cross_entropy,accuracy = cnn(config)

def data_statistics(d):
    # Train statistics
    lx,ly = d.raw_labels.shape
    print ' number of samples ', lx
    au_stats = np.zeros((ly,6),dtype=int)
    for i in xrange(lx):
        for j in xrange(ly):
            au_stats[j,int(d.raw_labels[i,j])] += 1

    for i in xrange(ly):
        print i,au_stats[i,:]

print 'train'
print '########################'
data_statistics(data.train)
print 'validation'
print '########################'
data_statistics(data.validation)
print 'test'
print '########################'
data_statistics(data.test)


saver = tf.train.Saver()

def vecprint(x,r):
    for i in x:
        n = round(i,r)
        if n == 0.0:
            print '   ',
        else:
            print n,
    print

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # Here's where you're restoring the variables w and b.
    # Note that the graph is exactly as it was when the variables were
    # saved in a prior training run.
    ckpt = tf.train.get_checkpoint_state(join(path,'model'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise Exception('Can\'t find checkpoint!')

    # batch_x, batch_y = data.validation.next_batch(config['batch_size'])
    # lx,ly = batch_y.shape
    # y_out = sess.run(y_conv, feed_dict={x:batch_x, y_:expand_labels(batch_y),keep_prob : 1.0})
    # for i in xrange(lx):
    #     print '------------------------------'
    #     vecprint(batch_y[i],1)
    #     vecprint(y_out[i],1)
