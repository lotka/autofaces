#!/usr/bin/python
import etc
import tensorflow as tf
reload(tf)
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
import sys
import math
from os.path import join

def conv_vis(i,sess,hconv,w,path,x,batch_x,keep_prob):
    # to visualize 1st conv layer Weights
    vv1 = sess.run(w)

    # to visualize 1st conv layer output
    vv2 = sess.run(hconv,feed_dict = {x:batch_x, keep_prob: 1.0})
    vv2 = vv2[0,:,:,:]   # in case of bunch out - slice first img


    def vis_conv(v,ix,iy,ch,cy,cx, p = 0) :
        v = np.reshape(v,(iy,ix,ch))
        # v = v[:,:,0,:]
        ix += 2
        iy += 2
        npad = ((1,1), (1,1), (0,0))
        v = np.pad(v, pad_width=npad, mode='constant', constant_values=p)
        v = np.reshape(v,(iy,ix,cy,cx))
        v = np.transpose(v,(2,0,3,1)) #cy,iy,cx,ix
        v = np.reshape(v,(cy*iy,cx*ix))
        return v

    # W_conv1 - weights
    shape = vv1.shape
    ix = int(shape[0])
    iy = int(shape[1])
    cs = int(shape[2])
    ch = int(shape[3])
    cy = 8   # grid from channels:  32 = 4x8
    cx = 8
    # print ix,iy,cs,ch,cy,cx
    v  = vis_conv(vv1,ix,iy,ch,cy,cx)
    plt.figure(figsize = (8,8))
    plt.imshow(v,cmap="Greys_r",interpolation='nearest')
    plt.colorbar()
    plt.savefig(join(path,'images/conv_weights_'+prefix(i,4)+'.png'),dpi=400)

    #  h_conv1 - processed image
    # print vv2.shape
    ix = vv2.shape[0]
    iy = vv2.shape[1]
    ch = vv2.shape[2]
    v  = vis_conv(vv2,ix,iy,ch,cy,cx)
    plt.figure(figsize = (8,8))
    plt.imshow(v,cmap="Greys_r",interpolation='nearest')
    plt.colorbar()
    plt.savefig(join(path,'images/conv_out_'+prefix(i,4)+'.png'),dpi=400)

    plt.close('all')


def prefix(i,zeros):
    s = str(i)
    while(len(s) < zeros):
        s = '0' + s
    return s

def main(data,config):

    def weight_variable(shape,name):
        conf = config['convolutions']
        if conf['weights_start_type'] == 'range':
            a,b = conf['weights_uniform_range']
            initial = tf.random_uniform(shape,a,b,name=name)
        elif conf['weights_start_type'] == 'constant':
            initial = tf.constant(conf['weights_constant'],shape=shape)
        elif conf['weights_start_type'] == 'std_dev':
            initial = tf.truncated_normal(shape, stddev=conf['weights_std_dev'])

        return tf.Variable(initial,name=name)

    def bias_variable(shape):
        # initial = tf.constant(0.1, shape=shape)
        # return tf.Variable(initial)
        start_value = config['convolutions']['bias_start']
        return tf.Variable(tf.constant(start_value,shape=shape))

    def conv2d(x, W,padding='SAME'):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

    def max_pool_layer(x,ksize,strides,layer_name,padding):
        with tf.name_scope(layer_name):
            activations = tf.nn.max_pool(x, ksize=ksize,strides=strides, padding=padding)
            print layer_name +  ' with size: ', ksize, ' with stride ', strides
            print ' output : ', activations.get_shape()
            return activations

    def variable_summaries(var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.scalar_summary('sttdev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                name = layer_name + '/weights'
                weights = weight_variable([input_dim, output_dim],name)
                variable_summaries(weights,name)
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases, layer_name + '/biases')
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.histogram_summary(layer_name + '/pre_activations', preactivate)
            activations = act(preactivate, 'activation')
            tf.histogram_summary(layer_name + '/activations', activations)

            print layer_name +  ' Shape: ', weights.get_shape(), ' with bias ', biases.get_shape()
            shape = activations.get_shape()
            print ' output : ', shape

            return activations

    def put_kernels_on_grid (kernel, (grid_Y, grid_X), pad=1):
        '''Visualize conv. features as an image (mostly for the 1st layer).
        Place kernel into a grid, with some paddings between adjacent filters.
        Args:
          kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
          (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                               User is responsible of how to break into two multiples.
          pad:               number of black pixels around each filter (between them)

        Return:
          Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
        '''
        # pad X and Y
        x1 = tf.pad(kernel, tf.constant( [[pad,0],[pad,0],[0,0],[0,0]] ))

        # X and Y dimensions, w.r.t. padding
        Y = kernel.get_shape()[0] + pad
        X = kernel.get_shape()[1] + pad
        NumChannels = int(kernel.get_shape()[2])

        # put NumKernels to the 1st dimension
        x2 = tf.transpose(x1, (3, 0, 1, 2))
        # organize grid on Y axis

        # print [grid_X, Y * grid_Y, X, 3]
        # print kernel.get_shape()
        x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, NumChannels]))
        # switch X and Y axes
        x4 = tf.transpose(x3, (0, 2, 1, 3))
        # organize grid on X axis
        x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, NumChannels]))

        # back to normal order (not combining with the next step for clarity)
        x6 = tf.transpose(x5, (2, 1, 3, 0))

        # to tf.image_summary order [batch_size, height, width, channels],
        #   where in this case batch_size == 1
        x7 = tf.transpose(x6, (3, 0, 1, 2))

        # scale to [0, 1]
        x_min = tf.reduce_min(x7)
        x_max = tf.reduce_max(x7)
        x8 = (x7 - x_min) / (x_max - x_min)

        return x8

    def cnn_layer(input_tensor,convolution_shape, padding, layer_name, act=tf.nn.relu):
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                name = layer_name + '/weights'
                weights = weight_variable(convolution_shape,name)
                variable_summaries(weights,name)
            with tf.name_scope('biases'):
                biases = bias_variable([convolution_shape[-1]])
                variable_summaries(biases, layer_name + '/biases')
            with tf.name_scope('convolution'):
                preactivate = conv2d(input_tensor, weights,padding=padding) + biases
                tf.histogram_summary(layer_name + '/pre_activations', preactivate)
            activations = act(preactivate, 'activation')
            tf.histogram_summary(layer_name + '/activations', activations)

            print layer_name +  ' Shape: ', weights.get_shape(), ' with bias ', biases.get_shape(), ' padding', padding
            shape = activations.get_shape()
            print ' output : ', shape

            channels = int(convolution_shape[-1])
            img_size = int(shape[-2])
            ix = img_size
            iy = img_size
            print channels,img_size

            cx = 0
            cy = 0
            if channels==32:
                cx = 8
                cy = 4
            if channels==64:
                cx = 8
                cy = 8
            if channels==128:
                cx = 16
                cy = 8

            assert channels == cx*cy
            assert cx != 0 and cy != 0

            if False:
                ## Prepare for visualization
                # Take only convolutions of first image, discard convolutions for other images.
                V = tf.slice(activations, (0, 0, 0, 0), (1, -1, -1, -1), name='slice_'+layer_name)
                V = tf.reshape(V, (img_size, img_size, channels))

                # Reorder so the channels are in the first dimension, x and y follow.
                V = tf.transpose(V, (2, 0, 1))
                # Bring into shape expected by image_summary
                V = tf.reshape(V, (-1, img_size, img_size, 1))
                tf.image_summary('a_'+ layer_name, V)

            if False:
                V = tf.slice(activations,(0,0,0,0),(1,-1,-1,-1))
                V = tf.reshape(V,(iy,ix,channels))

                ix = ix + 4
                iy = iy + 4

                V = tf.image.resize_image_with_crop_or_pad(V, iy, ix)

                V = tf.reshape(V,(iy,ix,cy,cx))
                V = tf.transpose(V,(2,0,3,1)) #cy,iy,cx,ix
                # V = np.einsum('yxYX->YyXx',V)
                # image_summary needs 4d input
                V = tf.reshape(V,[1,cy*iy,cx*ix,1])
                tf.image_summary('b_'+ layer_name, V)

            # if layer_name == 'Convolution_1':
            #     grid = put_kernels_on_grid(weights, (8, 8),pad=2)
            #     tf.image_summary('c_'+ layer_name, grid)
            # if layer_name == 'Convolution_2':
            #     grid = put_kernels_on_grid(weights, (8, 8),pad=1)
            #     tf.image_summary(layer_name + '/features', grid)
            # if layer_name == 'Convolution_3':
            #     grid = put_kernels_on_grid(weights, (8, 16),pad=1)
            #     tf.image_summary(layer_name + '/features', grid)

            return activations,weights
    input_dim = config['data']['image_shape']
    output_dim = config['data']['label_size']
    if type(input_dim) == type([]):
        shape_1 = [None, input_dim[0],input_dim[1]]
        shape_2 = [-1,input_dim[0],input_dim[1],1]
    else:
        shape_1 = [None, input_dim]
        shape_2 = [-1,int(np.sqrt(input_dim)),int(np.sqrt(input_dim)),1]
    x = tf.placeholder(np.float32, shape=shape_1,name='input')
    x_ = tf.reshape(x, shape=shape_2)
    y_ = tf.placeholder(np.float32, shape=[None, output_dim])

    print 'Input: ', x_.get_shape()
    print 'Outut: ', y_.get_shape()

    if config['network'] == 'convolution_gudi_2015':
        h_conv1,w1 = cnn_layer(x_,[5, 5, 1, 64], 'VALID', 'Convolution_1')
        # h_pool1 = max_pool_layer(h_conv1, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1],padding='SAME',layer_name='Max_Pool_1')
        # h_conv2,_ = cnn_layer(h_pool1,[5, 5, 64, 64], 'VALID', 'Convolution_2')
        # h_conv3,_ = cnn_layer(h_conv2,[4, 4, 64, 128], 'VALID', 'Convolution_3')

        with tf.name_scope('flatten'):
            prev_shape = h_conv1.get_shape()
            s = int(prev_shape[1])*int(prev_shape[2])*int(prev_shape[3])
            flat_1 = tf.reshape(h_conv1, [-1,s])
            print ' flatten to ', flat_1.get_shape()
    elif config['network'] == 'fullyconnected_subnetwork_gudi_2015':
        with tf.name_scope('flatten'):
            prev_shape = x_.get_shape()
            s = int(prev_shape[1])*int(prev_shape[2])*int(prev_shape[3])
            print s
            print ' flatten to ', s
            flat_1 = tf.reshape(x_, [-1,s])
    else:
        print 'Network', config['network'], 'not known.'
        exit()

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        flat_1_dropped = tf.nn.dropout(flat_1, keep_prob)

    h_full_1 = nn_layer(flat_1_dropped,s,100,'fully_connected_1')
    y_conv_unweighted = nn_layer(h_full_1,100,output_dim,'output',act=tf.nn.softmax)
    if config['ignore_empty_labels']:
        with tf.name_scope('sparse_weights'):
            y_conv = tf.transpose(tf.mul(tf.reduce_sum(tf.cast(y_,tf.float32),1),tf.transpose(y_conv_unweighted)))
    else:
        y_conv = y_conv_unweighted

    print 'Output shape : ', y_conv.get_shape()


    THRESHOLD = config['threshold']
    """
    Accuracy dodgyness
    """
    with tf.name_scope('accuracy_madness'):
        a = tf.cast(tf.greater(y_conv,THRESHOLD),tf.float32)
        b = tf.cast(tf.greater(y_,THRESHOLD),tf.float32)
        c = tf.abs(a-b)
        ab_sum = tf.reduce_sum(a,1)+tf.reduce_sum(b,1)
        accuracy = tf.reduce_mean(tf.Variable(1.0)-tf.reduce_sum(c,1)/ab_sum)

    with tf.name_scope('train'):
        with tf.name_scope('lmsq_loss'):
            lmsq_loss = tf.sqrt(tf.reduce_mean(tf.square(y_conv-y_)))
            tf.scalar_summary('loss/lmsq_loss', lmsq_loss)
        with tf.name_scope('cross_entropy'):
            cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
            tf.scalar_summary('loss/cross entropy', cross_entropy)

        r = config['learning_rate']
        if config['cost_function'] == 'cross_entropy':
            loss = cross_entropy
        else:
            loss = lmsq_loss

        if config['optimizer'] == 'adam':
            train_step = tf.train.AdamOptimizer(r).minimize(loss)
        else:
            train_step = tf.train.GradientDescentOptimizer(r).minimize(loss)

    if socket.gethostname() == 'ux305':
        sess = tf.Session()
    else:
        tensorflow_config = tf.ConfigProto()
        tensorflow_config.gpu_options.allow_growth=True
        sess = tf.Session(config=tensorflow_config)

    path = config.get_path()
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(join(path,'train'),sess.graph)
    validation_writer = tf.train.SummaryWriter(join(path,'validation'))

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

    N = config['iterations']
    random_batch = config['batch_randomisation']
    batch_size = config['batch_size']
    dropout_rate = config['dropout_rate']
    test_period = 10
    nN = N/test_period
    x_axis = np.zeros(nN)
    lmsq_axis = np.zeros((2,nN))
    cent_axis = np.zeros((2,nN))
    accu_axis = np.zeros((2,nN))

    train_auac_axis = np.zeros((nN,output_dim,4))
    validation_auac_axis = np.zeros((nN,output_dim,4))
    train_confusion = []
    validation_confusion = []

    def nice_seconds(t):
        if t < 60.0:
            result = str(round(t,1)) + ' seconds'
        elif t < 60**2:
            result = str(round(t/60.0,1)) + ' minutes'
        elif t < 60**3:
            result = str(round(t/(60.0**2),1)) + ' hours'
        else:
            result = str(t/(round(24.0*60.0**2,1))) + ' days'
        return result

    for i, pi in etc.range(N,info_frequency=5):
        if pi:
            e = pi.elapsed_time
            r = pi.time_remaining()
            print  i,'/',N,'Time elapsed:', nice_seconds(e), 'time remaining:', nice_seconds(r)

        batch_x, batch_y = data.train.next_batch(batch_size)
        train = {x: batch_x, y_: batch_y, keep_prob : dropout_rate}
        _train = {x: batch_x, y_: batch_y, keep_prob : 1.0}
        if (i+1) % test_period == 0:
            j = i/test_period
            vbatch_x, vbatch_y = data.validation.next_batch(batch_size)
            _validation = {x: vbatch_x, y_: vbatch_y, keep_prob : 1.0}

            x_axis[j] = i

            out = sess.run([lmsq_loss,cross_entropy,accuracy,y_conv], feed_dict=_validation)
            lmsq_axis[0,j],cent_axis[0,j],accu_axis[0,j],validation_y_out = out

            out = sess.run([lmsq_loss,cross_entropy,accuracy,y_conv], feed_dict=_train)
            lmsq_axis[1,j],cent_axis[1,j],accu_axis[1,j],train_y_out = out
            # train_writer.add_summary(summary, i)

            train_res, train_conf = metric.multi_eval(train_y_out,batch_y)
            train_auac_axis[j,:,:] = train_res

            validation_res, validation_conf = metric.multi_eval(validation_y_out,vbatch_y)
            validation_auac_axis[j,:,:] = validation_res

            train_confusion.append(train_conf)
            validation_confusion.append(validation_conf)

            if 'h_conv1' in locals():
                conv_vis(i,sess,h_conv1,w1,path,x,batch_x,keep_prob)

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            summary = sess.run(merged,
                               feed_dict=_train,
                               options=run_options,
                               run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%d' % i)
            train_writer.add_summary(summary, i)

            summary = sess.run(merged,
                               feed_dict=_validation,
                               options=run_options,
                               run_metadata=run_metadata)
            validation_writer.add_run_metadata(run_metadata, 'step%d' % i)
            validation_writer.add_summary(summary, i)
            print(config.get_path(), ' Adding run metadata for', i)
            print i,'   \t',
            print round(lmsq_axis[0,j],2),' ',
            print round(lmsq_axis[1,j],2),' ',
            print round(cent_axis[0,j],2),' ',
            print round(cent_axis[1,j],2),' '
        else:  # Record a summary
            sess.run(train_step, feed_dict=train)
            summary = sess.run(merged, feed_dict=_train)
            train_writer.add_summary(summary, i)



    thresholds = 20
    threshold_values = np.linspace(0,thresholds,thresholds)/float(thresholds)
    test_threshold_data = np.zeros((thresholds,output_dim,4))



    batches = int(data.test.labels.shape[0]/100)
    y_out = np.zeros((data.test.labels.shape[0],data.test.labels.shape[1]))
    for i in xrange(100):
        l = batches*i
        if i == 99:
            r = data.test.labels.shape[0]
        else:
            r = batches*(i+1)
        f = {x: data.test.images[l:r,:,:], y_: data.test.labels[l:r,:], keep_prob : 1.0}
        y_out[l:r,:] = sess.run(y_conv, feed_dict=f)

    for i in xrange(thresholds):
        results, confusion = metric.multi_eval(y_out,
                                               data.test.labels,
                                               threshold_values[i])
        test_threshold_data[i,:,:] = results


    """
    Save the model
    """
    saver = tf.train.Saver()
    p = join(config.get_path(),'model')
    if not os.path.isdir(p):
        os.mkdir(p)

    save_path = saver.save(sess, join(p,'model.ckpt'))
    print("Model saved in file: %s" % save_path)
    sess.close()

    """
    Save results
    """

    ssv_path = join(path,'numerical_data')
    if not os.path.isdir(ssv_path):
        os.mkdir(ssv_path)
    np.savetxt(join(ssv_path,'x_axis.ssv'),x_axis)
    np.savetxt(join(ssv_path,'lmsq.ssv'),lmsq_axis)
    np.savetxt(join(ssv_path,'cross_entropy.ssv'),cent_axis)
    np.savetxt(join(ssv_path,'naive_accuracy.ssv'),accu_axis)
    np.savez(join(ssv_path,'per_au_accuracy.npz'),
             train_metrics=train_auac_axis,
             validation_metrics=validation_auac_axis,
             train_confusion=train_confusion,
             validation_confusion=validation_confusion,
             threshold_values=threshold_values,
             test_threshold_data=test_threshold_data
             )


config =  PyExp(config_file='config/cnn.yaml',path='data')
if config['data']['dataset'] == 'disfa':
    import disfa
    data = disfa.Disfa(config['data'])
    config.update('data',data.config)

    if config['dump_frames']:
        im,lb = data.train.next_batch(1000)
        for i in xrange(10):
            plt.figure()
            # print i,im[i].shape[0]*im[i].shape[1], 48**2
            print i, lb[i]
            plt.imshow(im[i],vmax=10.0,vmin=-10.0)#,cmap="Greys_r",interpolation='nearest')
            plt.colorbar()
            plt.title(str(lb[i]))
            plt.savefig(join(config.get_path(),'images/face_'+str(i)+'.png'),dpi=100)
            plt.close('all')

    main(data,config)
else:
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    config['data']['image_shape'] = int(28**2)
    config['data']['label_size'] = int(10)
    main(mnist,config)

config.finished()
