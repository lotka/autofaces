#!/usr/bin/python
import tensorflow as tf
reload(tf)
import numpy as np
import matplotlib
matplotlib.use('Agg')
from expman import PyExp
import sys
from os.path import join
from model import cnn
import disfa
path = sys.argv[1]
# Load yaml config file
config = PyExp(config_file=join(path,'config.yaml'),make_new=False)
# Load data
data = disfa.Disfa(config['data'])
config.update('data',data.config,save=False)

# Load model
model = cnn(config)

x = model['x']
y_ = model['y']
train_step = model['train_step']
loss = model['loss']
y_conv = model['y_conv']
output_dim = model['output_dim']
keep_prob = model['keep_prob']
lmsq_loss = model['classifer_lmsq_loss']
cross_entropy = model['cross_entropy']
accuracy = model['accuracy']
alpha = model['alpha']
auto_loss = model['auto_loss']
mask = model['mask']
N = config['iterations']
batch_size = config['batch_size']
validation_batch_size = config['validation_batch_size']
dropout_rate = config['dropout_rate']

def data_statistics(d):
    # Train statistics
    lx,ly = d.raw_labels.shape
    print ' number of samples ', lx
    au_stats = np.zeros((ly,6),dtype=int)
    for i in xrange(lx):
        for j in xrange(ly):
            au_stats[j,int(d.raw_labels[i,j])] += 1

    for i in xrange(ly):
        print i,au_stats[i,:],float(au_stats[i,1:].sum())/float(au_stats[i,:1])

print 'train'
print '########################'
data_statistics(data.train)
print 'validation'
print '########################'
data_statistics(data.validation)
exit()

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
