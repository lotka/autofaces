#!/usr/bin/python
import matplotlib
import os
import socket
import sys

import numpy as np

import etc

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyexp import PyExp
from os.path import join
from sklearn.metrics import roc_curve, auc
from helper import get_all_experiments

def alpha_function(x,N):
    return 0.0

def alpha_constant(x,N,c=0.5):
    return c

def alpha_step(x,N,step=0.5):
    if float(x) > step*float(N):
        return 0.0
    else:
        return 1.0

def alpha_poly(_x,_N,order=1):
    x = float(_x)
    N = float(_N)
    return (1.0 - x/N)**order

def alpha_sigmoid(_x,_N,a=20,b=0.5):
    x = float(_x)
    N = float(_N)
    arg = x/N - b
    return 1.0/(1.0 + np.exp(a*arg))


def conv_vis(i, sess, hconv, w, path, x, batch_x, keep_prob):
    # to visualize 1st conv layer Weights
    vv1 = sess.run(w)

    # to visualize 1st conv layer output
    vv2 = sess.run(hconv, feed_dict={x: batch_x, keep_prob: 1.0})
    vv2 = vv2[0, :, :, :]  # in case of bunch out - slice first img

    def vis_conv(v, ix, iy, ch, cy, cx, p=0):
        v = np.reshape(v, (iy, ix, ch))
        # v = v[:,:,0,:]
        ix += 2
        iy += 2
        npad = ((1, 1), (1, 1), (0, 0))
        v = np.pad(v, pad_width=npad, mode='constant', constant_values=p)
        v = np.reshape(v, (iy, ix, cy, cx))
        v = np.transpose(v, (2, 0, 3, 1))  # cy,iy,cx,ix
        v = np.reshape(v, (cy * iy, cx * ix))
        return v

    # W_conv1 - weights
    shape = vv1.shape
    ix = int(shape[0])
    iy = int(shape[1])
    cs = int(shape[2])
    ch = int(shape[3])
    cy = 8  # grid from channels:  32 = 4x8
    cx = 8
    # print ix,iy,cs,ch,cy,cx
    v = vis_conv(vv1, ix, iy, ch, cy, cx)
    plt.figure(figsize=(8, 8))
    plt.imshow(v, cmap="Greys_r", interpolation='nearest')
    plt.colorbar()
    plt.savefig(join(path, 'images/conv_weights_' + prefix(i, 4) + '.png'), dpi=400)

    #  h_conv1 - processed image
    # print vv2.shape
    ix = vv2.shape[0]
    iy = vv2.shape[1]
    ch = vv2.shape[2]
    v = vis_conv(vv2, ix, iy, ch, cy, cx)
    plt.figure(figsize=(8, 8))
    plt.imshow(v, cmap="Greys_r", interpolation='nearest')
    plt.colorbar()
    plt.savefig(join(path, 'images/conv_out_' + prefix(i, 4) + '.png'), dpi=400)

    plt.close('all')


def prefix(i, zeros):
    s = str(i)
    while (len(s) < zeros):
        s = '0' + s
    return s


def save_model(sess, config, name):
    saver = tf.train.Saver()
    p = join(config.get_path(), 'models')
    if not os.path.isdir(p):
        os.mkdir(p)
    p = join(p, name)
    if not os.path.isdir(p):
        os.mkdir(p)

    save_path = saver.save(sess, join(p, 'model.ckpt'))
    print("Model saved in file: %s" % save_path)


def run(data, config):
    def put_kernels_on_grid(kernel, (grid_Y, grid_X), pad=1):
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
        x1 = tf.pad(kernel, tf.constant([[pad, 0], [pad, 0], [0, 0], [0, 0]]))

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

    def make_mask_batch(mean):
        mask_image = (mean > 0).astype(float)
        return np.expand_dims(np.array([mask_image for i in xrange(batch_size)]),axis=3)

    train_mask_batch      = make_mask_batch(data.train.mean_image)
    validation_mask_batch = make_mask_batch(data.validation.mean_image)

    # plt.figure()
    # plt.imshow(data.train.mean_image,interpolation='none')
    # plt.colorbar()
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(data.train.mean_image > 0,interpolation='none')
    # plt.colorbar()
    # plt.show()
    # exit()

    if socket.gethostname() == 'ux305':
        sess = tf.Session()
    else:
        tensorflow_config = tf.ConfigProto(allow_soft_placement=True)
        tensorflow_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tensorflow_config)

    path = config.get_path()
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(join(path, 'train'), sess.graph)
    validation_writer = tf.train.SummaryWriter(join(path, 'validation'))

    init = tf.initialize_all_variables()
    if config['seed_randomness']:
        tf.set_random_seed(config['seed'])
    sess.run(init)

    def roc_area(y_true, y_score, feed_dict):
        for i in xrange(y_true.shape[1]):
            trues = y_true[:, i]
            scores = sess.run(y_score, feed_dict)[:, i]
            if trues.sum() != 0.0:
                # print trues,scores
                fpr, tpr, t = roc_curve(trues, scores)
                print fpr, tpr, t
                roc_auc = auc(fpr, tpr)
                print roc_auc
            else:
                print '?',
        print



    if config['autoencoder']['function'] == 'constant':
        alpha_function = lambda x,N:  config['autoencoder']['constant']
    elif config['autoencoder']['function'] == 'step':
        alpha_function = lambda x, N: alpha_step(x, N, config['autoencoder']['step_percent'])
    elif config['autoencoder']['function'] == 'poly':
        alpha_function = lambda x, N: alpha_poly(x,N,config['autoencoder']['poly_order'])
    elif config['autoencoder']['function'] == 'sigmoid':
        alpha_function = lambda x, N: alpha_sigmoid(x,N)
    else:
        print 'INCORRECT ALPHA FUNCTION'
        exit()
    print 'Alpha ~ '
    for i in xrange(0,N,2):
        print alpha_function(i, N),
    print
    test_period = 10
    nN = N / test_period
    x_axis = np.zeros(nN)
    lmsq_axis = np.zeros((2, nN))
    cent_axis = np.zeros((2, nN))
    accu_axis = np.zeros((2, nN))
    auto_axis = np.zeros((2, nN))
    alpha_axis = np.zeros(nN)

    train_auac_axis = np.zeros((nN, output_dim, 4))
    validation_auac_axis = np.zeros((nN, output_dim, 4))
    train_confusion = []
    validation_confusion = []

    def nice_seconds(t):
        if t < 60.0:
            result = str(round(t, 1)) + ' seconds'
        elif t < 60 ** 2:
            result = str(round(t / 60.0, 1)) + ' minutes'
        elif t < 60 ** 3:
            result = str(round(t / (60.0 ** 2), 1)) + ' hours'
        else:
            result = str(t / (round(24.0 * 60.0 ** 2, 1))) + ' days'
        return result

    early_model_saved = False
    for i, pi in etc.range(N, info_frequency=10):
        if pi:
            e = pi.elapsed_time
            r = pi.time_remaining()
            print  i, '/', N,
            if config['data']['dataset'] == 'disfa':
                print ' | ',data.train.batch_counter,'/',data.train.nSamples,
            print ' | Time elapsed:', nice_seconds(e), 'time remaining:', nice_seconds(r)


        batch_x, batch_y = data.train.next_batch(batch_size)

        if config['binary_softmax']:
            train = {x: batch_x, y_: expand_labels(batch_y), keep_prob: dropout_rate, alpha : alpha_function(i,N), mask :  train_mask_batch}
            _train = {x: batch_x, y_: expand_labels(batch_y), keep_prob: 1.0, alpha : alpha_function(i,N) , mask :  train_mask_batch}
        else:
            train = {x: batch_x, y_: batch_y, keep_prob: dropout_rate, alpha : alpha_function(i,N) }
            _train = {x: batch_x, y_: batch_y, keep_prob: 1.0, alpha : alpha_function(i,N) }

        if (i + 1*0) % test_period == 0:
            j = i / test_period

            x_axis[j] = i

            # if 'h_conv1' in locals():
            #     conv_vis(i,sess,h_conv1,w1,path,x,batch_x,keep_prob)

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            summary = sess.run(merged,
                               feed_dict=_train,
                               options=run_options,
                               run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%d' % i)
            train_writer.add_summary(summary, i)

            """
            Get stuff for graphs
            """

            """
            Validation stuff
            """

            lmsq = 0.0
            cent = 0.0
            accu = 0.0
            auto = 0.0
            validation_y_out = np.zeros((validation_batch_size,output_dim))

            assert validation_batch_size/batch_size != 0, 'validation/train batch size must be an integer'

            for k in xrange(validation_batch_size/batch_size):
                vbatch_x, vbatch_y = data.validation.next_batch(batch_size,part=k,parts=validation_batch_size/batch_size)
                if config['binary_softmax']:
                    _validation = {x: vbatch_x, y_: expand_labels(vbatch_y), keep_prob: 1.0, alpha: alpha_function(i,N), mask :  validation_mask_batch}
                else:
                    _validation = {x: vbatch_x, y_: vbatch_y, keep_prob: 1.0, alpha: alpha_function(i,N), mask :  validation_mask_batch}

                out = sess.run([lmsq_loss, cross_entropy, accuracy,auto_loss, y_conv], feed_dict=_validation)
                _lmsq,_cent,_accu,_auto, _validation_y_out = out
                lmsq_axis[0, j] += _lmsq/float(validation_batch_size/batch_size)
                cent_axis[0, j] += _cent/float(validation_batch_size/batch_size)
                accu_axis[0, j] += _accu/float(validation_batch_size/batch_size)
                auto_axis[0, j] += _auto/float(validation_batch_size/batch_size)
                validation_y_out[batch_size * k:batch_size * (k + 1), :] = _validation_y_out

                # summary = sess.run(merged,
                #                    feed_dict=_validation,
                #                    options=run_options,
                #                    run_metadata=run_metadata)
                # validation_writer.add_run_metadata(run_metadata, 'step%d%d' % (i,k))
                # validation_writer.add_summary(summary, i)

            """
            """

            out = sess.run([lmsq_loss, cross_entropy, accuracy, auto_loss, y_conv], feed_dict=_train)
            lmsq_axis[1, j], cent_axis[1, j], accu_axis[1, j], auto_axis[1,j], train_y_out = out
            # train_writer.add_summary(summary, i)

            train_res, train_conf, _ = metric.multi_eval(train_y_out, batch_y)
            train_auac_axis[j, :, :] = train_res

            validation_res, validation_conf, _ = metric.multi_eval(validation_y_out, data.validation.next_batch(validation_batch_size)[1])
            validation_auac_axis[j, :, :] = validation_res

            train_confusion.append(train_conf)
            validation_confusion.append(validation_conf)

            # print(config.get_path(), ' Adding run metadata for', i)
            print  i, '/', N, ' | ',
            if config['data']['dataset'] == 'disfa':
                print data.train.batch_counter,'/',data.train.nSamples,' | ',
            print round(lmsq_axis[0, j], 5), ' ',
            print round(lmsq_axis[1, j], 5), ' ',
            print round(cent_axis[0, j], 5), ' ', #valid
            print round(cent_axis[1, j], 5), ' ', #train
            print round(auto_axis[0, j], 5), ' ', #valid
            print round(auto_axis[1, j], 5), ' ', #train
            alpha_axis[j] = alpha_function(i,N)
            print alpha_function(i,N), ' ', #alpha
            if early_model_saved:
                print '(early saved)'
            else:
                print
            if not early_model_saved:
                # early_condition_1 = (cent_axis[1, :] < cent_axis[0, :]).sum() > 2
                early_condition_2 = float(i) >= float(N)*config.config['global']['early_stop_percent']
                if early_condition_2:
                    # Record the number at which the early model is saved
                    config.config['results']['early_stop_iteration'] = int(x_axis[j])
                    # Save it
                    save_model(sess, config, 'early')
                    early_model_saved = True


        else:  # Record a summary
            sess.run(train_step, feed_dict=train)
            summary = sess.run(merged, feed_dict=_train)
            train_writer.add_summary(summary, i)

    # print 'Test set analysis'
    #
    # thresholds = 20
    # padding = 0.1
    # threshold_values = np.linspace(-padding,thresholds+padding,thresholds)/float(thresholds)
    # test_threshold_data = np.zeros((thresholds,output_dim,4))
    #
    # nBatches = int(100)
    # batches = int(data.test.labels.shape[0]/nBatches)
    # y_out = np.zeros((data.test.labels.shape[0],data.test.labels.shape[1]))
    #
    #
    # for i in tqdm(xrange(nBatches)):
    #     l = batches*i
    #     if i == nBatches-1:
    #         r = data.test.labels.shape[0]
    #     else:
    #         r = batches*(i+1)
    #     f = {x: data.test.images[l:r], keep_prob : 1.0}
    #     y_out[l:r] = sess.run(y_conv, feed_dict=f)
    #     y_true = data.test.labels
    #
    # test_confusion = []
    # for i in xrange(thresholds):
    #     results, confusion = metric.multi_eval(y_out,
    #                                            y_true,
    #                                            threshold_values[i])
    #     test_confusion.append(confusion)
    #     test_threshold_data[i,:,:] = results


    """
    Save the model
    """
    save_model(sess, config, 'final')

    """
    Close the session
    """

    sess.close()

    """
    Save results
    """

    ssv_path = join(path, 'numerical_data')
    if not os.path.isdir(ssv_path):
        os.mkdir(ssv_path)
    np.savetxt(join(ssv_path, 'x_axis.ssv'), x_axis)
    np.savetxt(join(ssv_path, 'lmsq.ssv'), lmsq_axis)
    np.savetxt(join(ssv_path, 'autoloss.ssv'), auto_axis)
    np.savetxt(join(ssv_path, 'alpha_axis.ssv'), alpha_axis)
    np.savetxt(join(ssv_path, 'cross_entropy.ssv'), cent_axis)
    np.savetxt(join(ssv_path, 'naive_accuracy.ssv'), accu_axis)
    np.savez(join(ssv_path, 'per_au_accuracy.npz'),
             train_metrics=train_auac_axis,
             validation_metrics=validation_auac_axis,
             train_confusion=train_confusion,
             validation_confusion=validation_confusion
             )


def main(path,config_file,config_overwrite=None):
    if path == None:
        if socket.gethostname() == 'ux305':
            path = '/home/luka/Documents/autofaces/data'
        else:
            path = '/vol/lm1015-tmp/data/'

    config = PyExp(config_file=config_file,
                   path=path,
                   config_overwrite=config_overwrite)

    if config['data']['dataset'] == 'disfa':
        import disfa

        data = disfa.Disfa(config['data'])
        config.update('data', data.config)

        im, lb = data.train.next_batch(5)
        for i in xrange(5):
            plt.figure()
            # print i,im[i].shape[0]*im[i].shape[1], 48**2
            print i, lb[i]
            if config['data']['scaling'] == '[-1,1]':
                plt.imshow(im[i], vmax=1.0, vmin=-1.0)
            elif config['data']['scaling'] == '[0,1]':
                plt.imshow(im[i], vmax=1.0, vmin=0.0)
            else:
                plt.imshow(im[i])
            plt.colorbar()
            plt.title(str(lb[i]))
            plt.savefig(join(config.get_path(), 'images/face_' + str(i) + '.png'), dpi=100)
            plt.close('all')

        run(data, config)
    else:
        from tensorflow.examples.tutorials.mnist import input_data

        data = input_data.read_data_sets("MNIST_data/", one_hot=True)
        config['data']['image_shape'] = int(28 ** 2)
        config['data']['label_size'] = int(10)
        run(data, config)

    return config.finished(), data

def run_experiment(args,config_overwrite=None):

    tf.reset_default_graph()
    with tf.device('/' + args.device + ':0'):
        data_path,data = main(args.path,args.config_file,config_overwrite=config_overwrite)

    import test_set_analysis

    if args.device == None:
        args.device = 'gpu'

    data.validation.batch_counter = 0
    tf.reset_default_graph()
    test_set_analysis.main((sys.argv[0], data_path,'final', args.device), data=data, overwrite_dict=config_overwrite)

    data.validation.batch_counter = 0
    tf.reset_default_graph()
    test_set_analysis.main((sys.argv[0], data_path, 'early', args.device), data=data, overwrite_dict=config_overwrite)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', action='store', dest='device',help='Store a simple value')
    parser.add_argument('--path', action='store', dest='path',help='Store a simple value')
    parser.add_argument('--config', action='store', dest='config_file', help='Store a simple value',type=str)
    parser.add_argument('--batch', action='store', dest='batch', help='Store a simple value',type=int)
    parser.add_argument('--compare', action='store', dest='compare' ,help='compare different configs',type=bool)

    args = parser.parse_args()

    if args.device == None:
        print 'Error. Program requires device flag to be cpu or gpu.'
        exit()

    from model import cnn, expand_labels, tf
    import metric

    o = {}
    if args.compare == True and args.compare != None:
        if args.config_file == 'config/auto.yaml':
            o['data:normalisation']  = ['contrast_[-inf,inf]',
                                        'contrast_[-1,1]',
                                        'face_[-inf,inf]',
                                        'face_[-1,1]',
                                        'none_[0,1]']
            # o['autoencoder:activation'] = ['tanh','sigmoid','linear','relu']
        if args.config_file == 'config/class.yaml':
            o['data:normalisation'] = ['contrast_[-inf,inf]',
                                       'contrast_[-1,1]',
                                       'face_[-inf,inf]',
                                       'face_[-1,1]',
                                       'none_[0,1]'][::-1]
    # print args.batch
    # o['autoencoder:activation'] =  [args.batch]
    overwrite_dicts = get_all_experiments(o)
    for i,x in enumerate(overwrite_dicts):
        print i+1,x
    # raw_input('any key plz')
    if o == None:
        run_experiment(args)
    else:
        for exp in overwrite_dicts:
            run_experiment(args, config_overwrite=exp)

