#!/usr/bin/python
import matplotlib

import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from helper import get_n_idx_biggest, get_n_idx_smallest, get_n_idx_near_mean

def test_model(name,data,config,path):


    if socket.gethostname() == 'ux305':
        sess = tf.Session()
    else:
        tensorflow_config = tf.ConfigProto(allow_soft_placement=True)
        tensorflow_config.gpu_options.allow_growth=True
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

    x = model['x']
    y_ = model['y']
    y_conv = model['y_conv']
    output_dim = model['output_dim']
    keep_prob = model['keep_prob']
    alpha = model['alpha']
    auto_loss = model['auto_loss']
    batch_size = int(model['batch_size'])
    y_image = model['y_image']
    mask = model['mask']

    def make_mask_batch(mean):
        mask_image = (mean > 50).astype(float)
        return np.expand_dims(np.array([mask_image for i in xrange(batch_size)]),axis=3)

    mask_batch = make_mask_batch(data.validation.mean_image)


    thresholds = 20
    padding = 0.1
    threshold_values = np.linspace(-padding, thresholds+padding, thresholds)/float(thresholds)
    test_threshold_data = np.zeros((thresholds,output_dim,4))

    nBatches = int(float(data.validation.labels.shape[0])/float(batch_size))
    y_out = np.zeros(data.validation.labels.shape)
    autoencoder_loss = 0.0
    true_autoencoder_loss = 0.0

    a = data.validation.next_batch(100)[0]
    b = sess.run(model['y_image'],feed_dict={x: data.validation.next_batch(100)[0], keep_prob: 1.0,alpha: 1.0, mask : mask_batch})[:,:,:,0]
    validation_images = (data.validation.inverse_process(a),data.validation.inverse_process(b))
    # validation_images = (a,b)

    a = data.train.next_batch(100)[0]
    b = sess.run(model['y_image'],feed_dict={x: data.train.next_batch(100)[0],keep_prob: 1.0, alpha: 1.0, mask : mask_batch})[:,:,:,0]
    train_images = (data.train.inverse_process(a),data.train.inverse_process(b))
    # train_images = (a,b)

    print 'VALIDATION: Getting autoencoder loss..'
    true_losses = np.array([])
    for i in tqdm(xrange(nBatches)):
        l = batch_size*i
        r = batch_size*(i+1)
        current_batch = data.validation.images[l:r]

        y_out[l:r] = sess.run(y_conv, feed_dict={x: current_batch, keep_prob : 1.0, alpha : 0.0, mask : mask_batch})
        autoencoder_loss += sess.run(auto_loss, feed_dict={x: current_batch, mask : mask_batch})/float(nBatches)
        # calculate true loss
        _in  = data.validation.images[l:r,:,:]
        _out = sess.run(y_image, feed_dict={x: current_batch, mask : mask_batch})[:,:,:,0]
        true_losses = np.append(true_losses,data.validation.true_loss(_in,_out,base=l))



    idx_big   = get_n_idx_biggest(true_losses, 100)
    idx_small = get_n_idx_smallest(true_losses, 100)
    idx_mean  = get_n_idx_near_mean(true_losses, 100)
    print 'idx_big'
    print idx_big
    print 'idx_small'
    print idx_small
    print 'idx_mean'
    print idx_mean
    assert len(idx_big) == 100
    assert len(idx_small) == 100
    assert len(idx_mean) == 100

    i_big   = data.validation.images[idx_big,   :,:]
    i_small = data.validation.images[idx_small, :, :]
    i_mean  = data.validation.images[idx_mean,  :, :]
    i_big_original   = data.validation.images_original[idx_big,   :,:]
    i_small_original = data.validation.images_original[idx_small, :, :]
    i_mean_original  = data.validation.images_original[idx_mean,  :, :]

    o_big   = sess.run(model['y_image'], feed_dict={x: i_big,   keep_prob: 1.0, alpha: 1.0, mask : mask_batch})[:, :, :, 0]
    o_small = sess.run(model['y_image'], feed_dict={x: i_small, keep_prob: 1.0, alpha: 1.0, mask : mask_batch})[:, :, :, 0]
    o_mean  = sess.run(model['y_image'], feed_dict={x: i_mean,  keep_prob: 1.0, alpha: 1.0, mask : mask_batch})[:, :, :, 0]

    i_big       = (i_big_original, i_big.copy(),       data.validation.inverse_process(i_big.copy()))
    i_small     = (i_small_original, i_small.copy(),     data.validation.inverse_process(i_small.copy()))
    i_mean      = (i_mean_original, i_mean.copy(),      data.validation.inverse_process(i_mean.copy()))
    label_big   = data.validation.labels[idx_big,:]
    label_small = data.validation.labels[idx_small,:]
    label_mean  = data.validation.labels[idx_mean,:]
    o_big       = (o_big.copy()     ,  data.validation.inverse_process(o_big.copy()))
    o_small     = (o_small.copy()   ,  data.validation.inverse_process(o_small.copy()))
    o_mean      = (o_mean.copy()    ,  data.validation.inverse_process(o_mean.copy()))

    plt.figure()
    plt.imshow(i_big[0][0])
    plt.colorbar()
    plt.show()

    true_autoencoder_loss = true_losses.mean()

    y_out = y_out[:nBatches*(batch_size+1),:]
    data.validation.labels = data.validation.labels[:nBatches*(batch_size+1),:]


    print 'VALIDATION: Comparing threshold values...'
    test_confusion = []
    test_roc_data = []
    for i in tqdm(xrange(thresholds)):
        results, confusion, roc_data = metric.multi_eval(y_out,
                                                         data.validation.labels,
                                                         threshold_values[i])
        test_confusion.append(confusion)
        test_roc_data.append(roc_data)
        test_threshold_data[i,:,:] = results

    ssv_path = join(path,'numerical_data')
    if not isdir(ssv_path):
        mkdir(ssv_path)
    print 'VALIDATION: Saving results..'
    np.savez(join(ssv_path,name+'_model_analysis.npz'),
             threshold_values=threshold_values,
             test_threshold_data=test_threshold_data,
             test_confusion=test_confusion,
             test_roc_data=test_roc_data,
             autoencoder_loss=autoencoder_loss,
             true_autoencoder_loss=true_autoencoder_loss,
             true_losses=true_losses,
             i_big=i_big,
             i_small=i_small,
             i_mean=i_mean,
             label_big=label_big,
             label_small=label_small,
             label_mean=label_mean,
             o_big=o_big,
             o_small=o_small,
             o_mean=o_mean,
             auto_images=(train_images,validation_images))

    plt.figure()
    v = np.load(join(ssv_path, name + '_model_analysis.npz'))['i_big'][0,0]
    plt.imshow(v)
    plt.colorbar()
    plt.show()

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
