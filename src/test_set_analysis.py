#!/usr/bin/python
import matplotlib

import numpy as np

matplotlib.use('Agg')
from pyexp import PyExp
import sys
import metric
import socket
from os.path import join
from os.path import isdir
from os import mkdir
from tqdm import tqdm
from model import cnn
import disfa
import tensorflow as tf

def thresholding(tf,sess,data,model,config):

    x = model['x']
    y_ = model['y']
    y_conv = model['y_conv']
    output_dim = model['output_dim']
    keep_prob = model['keep_prob']
    alpha = model['alpha']
    auto_loss = model['auto_loss']
    batch_size = int(model['batch_size'])


    thresholds = 20
    padding = 0.1
    threshold_values = np.linspace(-padding, thresholds+padding, thresholds)/float(thresholds)
    test_threshold_data = np.zeros((thresholds,output_dim,4))

    nBatches = int(float(data.validation.labels.shape[0])/float(batch_size))
    y_out = np.zeros((data.validation.labels.shape[0],data.validation.labels.shape[1]))
    autoencoder_loss = 0.0

    validation_images = (data.validation.next_batch(100)[0], sess.run(model['y_image'],feed_dict={x: data.validation.next_batch(100)[0], keep_prob: 1.0,alpha: 1.0})[:,:,:,0])
    train_images = (data.train.next_batch(100)[0], sess.run(model['y_image'],feed_dict={x: data.train.next_batch(100)[0],keep_prob: 1.0, alpha: 1.0})[:,:,:,0])

    for i in tqdm(xrange(nBatches)):
        l = batch_size*i
        r = batch_size*(i+1)
        # if i == nBatches-1:
        #     r = data.validation.labels.shape[0]
        # else:
        #
        y_out[l:r] = sess.run(y_conv, feed_dict={x: data.validation.images[l:r], keep_prob : 1.0, alpha : 0.0})
        autoencoder_loss += sess.run(auto_loss, feed_dict={x: data.validation.images[l:r]})/float(nBatches)
        y_true = data.validation.labels

    test_confusion = []
    test_roc_data = []
    for i in xrange(thresholds):
        results, confusion, roc_data = metric.multi_eval(y_out,
                                               y_true,
                                               threshold_values[i])
        test_confusion.append(confusion)
        test_roc_data.append(roc_data)
        test_threshold_data[i,:,:] = results

    return threshold_values,test_threshold_data, test_confusion, test_roc_data, autoencoder_loss, (train_images,validation_images)

def test_model(name,data,config,path):

    if socket.gethostname() == 'ux305':
        sess = tf.Session()
    else:
        tensorflow_config = tf.ConfigProto(allow_soft_placement=True)
        tensorflow_config.gpu_options.allow_growth=False
        sess = tf.Session(config=tensorflow_config)
    # Load model
    nBatches = int(500)
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

    threshold_values,test_threshold_data, test_confusion, test_roc_data, autoencoder_loss, auto_images = thresholding(tf,sess,data,model,config)

    ssv_path = join(path,'numerical_data')
    if not isdir(ssv_path):
        mkdir(ssv_path)

    np.savez(join(ssv_path,name+'_model_analysis.npz'),
             threshold_values=threshold_values,
             test_threshold_data=test_threshold_data,
             test_confusion=test_confusion,
             test_roc_data=test_roc_data,
             autoencoder_loss=autoencoder_loss,
             auto_images=auto_images)

    sess.close()

def main(argv,data=None,overwrite_dict=None):
    path = argv[1]
    model = argv[2]
    device = argv[3]
    with tf.device('/'+device+':0'):
        # Load yaml config file
        config = PyExp(config_file=join(path, 'config.yaml'), make_new=False,config_overwrite=overwrite_dict)
        # Load data
        if data == None:
            data = disfa.Disfa(config['data'])

        config.update('data', data.config, save=False)

        test_model(model,data,config,path)

if __name__ == "__main__":
    main(sys.argv)
