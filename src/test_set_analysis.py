#!/usr/bin/python
import etc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyexp.pyexp import PyExp
import math
import sys
import math
import metric
import socket
from os.path import join
from os.path import isdir
from os import mkdir
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

def thresholding(tf,sess,data,model):

    x,y_,train_step,loss, y_conv,output_dim,keep_prob,lmsq_loss,cross_entropy,accuracy = model

    thresholds = 20
    padding = 0.1
    threshold_values = np.linspace(-padding, thresholds+padding, thresholds)/float(thresholds)
    test_threshold_data = np.zeros((thresholds,output_dim,4))

    nBatches = int(100)
    batches = int(data.test.labels.shape[0]/nBatches)
    y_out = np.zeros((data.test.labels.shape[0],data.test.labels.shape[1]))


    for i in tqdm(xrange(nBatches)):
        l = batches*i
        if i == nBatches-1:
            r = data.test.labels.shape[0]
        else:
            r = batches*(i+1)
        f = {x: data.test.images[l:r], keep_prob : 1.0}
        y_out[l:r] = sess.run(y_conv, feed_dict=f)
        y_true = data.test.labels

    test_confusion = []
    test_roc_data = []
    for i in xrange(thresholds):
        results, confusion, roc_data = metric.multi_eval(y_out,
                                               y_true,
                                               threshold_values[i])
        test_confusion.append(confusion)
        test_roc_data.append(roc_data)
        test_threshold_data[i,:,:] = results

    return threshold_values,test_threshold_data, test_confusion, test_roc_data

def test_model(name,data):
    import tensorflow as tf
    if socket.gethostname() == 'ux305':
        sess = tf.Session()
    else:
        tensorflow_config = tf.ConfigProto()
        tensorflow_config.gpu_options.allow_growth=True
        sess = tf.Session(config=tensorflow_config)
    # Load model
    model = cnn(config)

    saver = tf.train.Saver()

    sess.run(tf.initialize_all_variables())
    # Here's where you're restoring the variables w and b.
    # Note that the graph is exactly as it was when the variables were
    # saved in a prior training run.
    print join(path,'models/'+name)
    ckpt = tf.train.get_checkpoint_state(join(path,'models/'+name))
    if ckpt and ckpt.model_checkpoint_path:
        print ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise Exception('Can\'t find checkpoint!')

    threshold_values,test_threshold_data, test_confusion, test_roc_data = thresholding(tf,sess,data,model)

    ssv_path = join(path,'numerical_data')
    if not isdir(ssv_path):
        mkdir(ssv_path)

    np.savez(join(ssv_path,name+'_model_analysis.npz'),
             threshold_values=threshold_values,
             test_threshold_data=test_threshold_data,
             test_confusion=test_confusion,
             test_roc_data=test_roc_data)

    sess.close()

test_model(sys.argv[2],data)
