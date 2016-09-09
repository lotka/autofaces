#!/usr/bin/python
import matplotlib

import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from expman import PyExp
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
    model = cnn(config,train=False)

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
    accuracy_data = np.zeros(thresholds)

    nBatches = int(float(data.validation.labels.shape[0])/float(batch_size))
    y_out = np.zeros(data.validation.labels.shape)
    autoencoder_loss = 0.0
    true_autoencoder_loss = 0.0

    a = data.validation.next_batch(batch_size)[0]
    b = sess.run(model['y_image'],feed_dict={x: data.validation.next_batch(batch_size)[0], keep_prob: 1.0,alpha: 1.0, mask : mask_batch})[:,:,:,0]
    validation_images = (data.validation.inverse_process(a),data.validation.inverse_process(b))
    # validation_images = (a,b)

    a = data.train.next_batch(batch_size)[0]
    b = sess.run(model['y_image'],feed_dict={x: data.train.next_batch(batch_size)[0],keep_prob: 1.0, alpha: 1.0, mask : mask_batch})[:,:,:,0]
    train_images = (data.train.inverse_process(a),data.train.inverse_process(b))
    # train_images = (a,b)

    print 'VALIDATION: Feeding validation set through network..'
    true_losses = np.array([])
    for i in tqdm(xrange(nBatches)):
        l = batch_size*i
        r = batch_size*(i+1)
        current_batch = data.validation.images[l:r]

        y_out[l:r] = sess.run(y_conv, feed_dict={x: current_batch, keep_prob : 1.0, alpha : 0.0, mask : mask_batch})
        autoencoder_loss += sess.run(auto_loss, feed_dict={x: current_batch, mask : mask_batch, keep_prob: 1.0})/float(nBatches)
        # calculate true loss
        _in  = data.validation.images[l:r,:,:]
        _out = sess.run(y_image, feed_dict={x: current_batch, mask : mask_batch, keep_prob: 1.0})[:,:,:,0]
        true_losses = np.append(true_losses,data.validation.true_loss(_in,_out,base=l))



    idx_big   = get_n_idx_biggest(true_losses, batch_size)
    idx_small = get_n_idx_smallest(true_losses, batch_size)
    idx_mean  = get_n_idx_near_mean(true_losses, batch_size)
    print 'idx_big'
    print idx_big
    print 'idx_small'
    print idx_small
    print 'idx_mean'
    print idx_mean
    assert len(idx_big) == batch_size
    assert len(idx_small) == batch_size
    assert len(idx_mean) == batch_size

    i_big   = data.validation.images[idx_big,   :,:]
    i_small = data.validation.images[idx_small, :, :]
    i_mean  = data.validation.images[idx_mean,  :, :]
    print data.validation.idx_interesting
    idx_interesting = list(data.validation.idx_interesting)
    print idx_interesting
    i_interesting = data.validation.images[idx_interesting,:,:]

    t_big = data.validation.labels[idx_big,:]
    t_small = data.validation.labels[idx_small,:]
    t_mean = data.validation.labels[idx_mean,:]
    t_interesting = data.validation.labels[idx_interesting,:]

    if hasattr(data.validation, 'images_original'):
        i_big_original   = data.validation.images_original[idx_big,   :,:]
        i_small_original = data.validation.images_original[idx_small, :, :]
        i_mean_original  = data.validation.images_original[idx_mean,  :, :]
        i_interesting_original  = data.validation.images_original[idx_interesting,  :, :]
    else:
        # Just so we don't always have to have these images
        i_big_original   = np.zeros(i_mean.shape)
        i_small_original = np.zeros(i_mean.shape)
        i_mean_original  = np.zeros(i_mean.shape)
        i_interesting_original = np.zeros(i_mean.shape)

    o_big,   p_big   = sess.run([model['y_image'],model['y_conv']], feed_dict={x: i_big,   keep_prob: 1.0, alpha: 1.0, mask : mask_batch})
    o_small, p_small = sess.run([model['y_image'],model['y_conv']], feed_dict={x: i_small, keep_prob: 1.0, alpha: 1.0, mask : mask_batch})
    o_mean,  p_mean  = sess.run([model['y_image'],model['y_conv']], feed_dict={x: i_mean,  keep_prob: 1.0, alpha: 1.0, mask : mask_batch})
    o_interesting,  p_interesting  = sess.run([model['y_image'],model['y_conv']], feed_dict={x: i_interesting,  keep_prob: 1.0, alpha: 1.0, mask : mask_batch})

    o_big   = o_big[:, :, :, 0]
    o_small = o_small[:, :, :, 0]
    o_mean  = o_mean[:, :, :, 0]
    o_interesting = o_interesting[:,:,:,0]

    i_big       = (i_big_original, i_big.copy(),       data.validation.inverse_process(i_big.copy(),idx=idx_big))
    i_small     = (i_small_original, i_small.copy(),     data.validation.inverse_process(i_small.copy(),idx=idx_small))
    i_mean      = (i_mean_original, i_mean.copy(),      data.validation.inverse_process(i_mean.copy(),idx=idx_mean))
    i_interesting      = (i_interesting_original, i_interesting.copy(),      data.validation.inverse_process(i_interesting.copy(),idx=idx_interesting))
    label_big   = data.validation.labels[idx_big,:]
    label_small = data.validation.labels[idx_small,:]
    label_mean  = data.validation.labels[idx_mean,:]
    label_interesting = data.validation.labels[idx_interesting,:]
    o_big       = (o_big.copy()     ,  data.validation.inverse_process(o_big.copy(),idx=idx_big))
    o_small     = (o_small.copy()   ,  data.validation.inverse_process(o_small.copy(),idx=idx_small))
    o_mean      = (o_mean.copy()    ,  data.validation.inverse_process(o_mean.copy(),idx=idx_mean))
    o_interesting      = (o_interesting.copy()    ,  data.validation.inverse_process(o_interesting.copy(),idx=idx_interesting))

    true_autoencoder_loss = true_losses.mean()

    y_out = y_out[:nBatches*(batch_size+1),:]
    data.validation.labels = data.validation.labels[:nBatches*(batch_size+1),:]


    print 'VALIDATION: Comparing threshold values...'
    test_confusion = []
    test_roc_data = []
    for i in tqdm(xrange(thresholds)):
        results, confusion, roc_data, accuracy_data[i] = metric.multi_eval(y_out,
                                                         data.validation.labels,
                                                         threshold_values[i])
        test_confusion.append(confusion)
        test_roc_data.append(roc_data)
        test_threshold_data[i,:,:] = results

    ssv_path = join(path,'numerical_data')
    if not isdir(ssv_path):
        mkdir(ssv_path)
    print 'VALIDATION: Saving results..'
    np.savez_compressed(join(ssv_path,name+'_model_analysis.npz'),
             threshold_values=threshold_values,
             test_threshold_data=test_threshold_data,
             test_confusion=test_confusion,
             test_roc_data=test_roc_data,
             accuracy_data=accuracy_data,
             autoencoder_loss=autoencoder_loss,
             true_autoencoder_loss=true_autoencoder_loss,
             true_losses=true_losses,
             i_big=i_big,
             i_small=i_small,
             i_mean=i_mean,
             i_interesting=i_interesting,
             t_big=t_big,
             t_small=t_small,
             t_mean=t_mean,
             t_interesting=t_interesting,
             label_big=label_big,
             label_small=label_small,
             label_mean=label_mean,
             label_interesting=label_interesting,
             p_big=p_big,
             p_small=p_small,
             p_mean=p_mean,
             p_interesting=p_interesting,
             o_big=o_big,
             o_small=o_small,
             o_mean=o_mean,
             o_interesting=o_interesting,
             idx_small=idx_small,
             idx_big=idx_big,
             idx_mean=idx_mean,
             idx_interesting=idx_interesting,
             valid_subject_idx=data.validation.subject_idx,
             train_subject_idx=data.train.subject_idx,
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
        if data is None:
            data = disfa.Disfa(config['data'])

        config.update('data', data.config, save=False)

        test_model(model,data,config,path)

if __name__ == "__main__":
    main(sys.argv)
