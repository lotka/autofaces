import numpy as np
import tensorflow as tf
from tqdm import tqdm


def batched_feed(data, nBatches, op, sess):
    batches = int(data.labels.shape[0] / nBatches)
    y_out = np.zeros((data.labels.shape[0], data.labels.shape[1]))

    for i in tqdm(xrange(nBatches)):

        l = batches * i
        if i == nBatches - 1:
            r = data.labels.shape[0]
        else:
            r = batches * (i + 1)

        y_out[l:r] = sess.run(op, feed_dict={x: data.images[l:r], keep_prob: 1.0})

    return y_out


def expand_labels(labels):
    lx, ly = labels.shape
    expanded_labels = np.zeros((lx, ly * 2))

    for i in xrange(lx):
        for j in xrange(ly):
            if labels[i, j] > 0:
                expanded_labels[i, 2 * j] = 1.0
                expanded_labels[i, 2 * j + 1] = 0.0
            else:
                expanded_labels[i, 2 * j] = 0.0
                expanded_labels[i, 2 * j + 1] = 1.0

    return expanded_labels


def weight_variable(shape, name, config):
    if config['seed_randomness']:
        seed = config['seed']
    else:
        seed = None

    conf = config['weights']
    if conf['weights_start_type'] == 'range':
        a, b = conf['weights_uniform_range']
        initial = tf.random_uniform(shape, a, b, name=name,seed=seed)
    elif conf['weights_start_type'] == 'constant':
        initial = tf.constant(conf['weights_constant'], shape=shape)
    elif conf['weights_start_type'] == 'std_dev':
        initial = tf.truncated_normal(shape, stddev=conf['weights_std_dev'],seed=seed)

    return tf.Variable(initial, name=name)


def bias_variable(shape, config):
    # initial = tf.constant(0.1, shape=shape)
    # return tf.Variable(initial)
    start_value = config['weights']['bias_start']
    return tf.Variable(tf.constant(start_value, shape=shape))


def conv2d(x, W, padding='SAME',strides=[1,1,1,1]):
    return tf.nn.conv2d(x, W, strides=strides, padding=padding)

def conv2d_transpose(x, W,output_shape, padding='SAME',strides=[1,1,1,1]):
    return tf.nn.conv2d_transpose(x, W,output_shape, strides=strides, padding=padding)


def pool_layer(type, x, ksize, strides, layer_name, padding):
    with tf.name_scope(layer_name):
        if type == 'max':
            pool = tf.nn.max_pool
        elif type == 'avg':
            pool = tf.nn.avg_pool
        activations = pool(x, ksize=ksize, strides=strides, padding=padding)
        print layer_name + ' with size: ', ksize, ' with stride ', strides
        print ' output : ', activations.get_shape()
        return (activations, None)


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


def nn_layer(input_tensor, input_dim, output_dim, layer_name, config, act=tf.nn.relu):
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            name = layer_name + '/weights'
            weights = weight_variable([input_dim, output_dim], name, config)
            variable_summaries(weights, name)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim], config)
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.histogram_summary(layer_name + '/pre_activations', preactivate)
        activations = act(preactivate, 'activation')
        tf.histogram_summary(layer_name + '/activations', activations)

        print layer_name + ' Shape: ', weights.get_shape(), ' with bias ', biases.get_shape()
        shape = activations.get_shape()
        print ' output : ', shape

        return activations,None

def dcnn_layer(input_tensor, convolution_shape,output_shape, padding, layer_name, config, act=tf.nn.relu,strides=[1,1,1,1]):
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            name = layer_name + '/weights'
            weights = weight_variable(convolution_shape, name, config)
            variable_summaries(weights, name)
        with tf.name_scope('biases'):
            biases = bias_variable([convolution_shape[-1]], config)
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('convolution'):
            preactivate = conv2d_transpose(input_tensor, weights,output_shape, padding=padding,strides=strides) + biases
            tf.histogram_summary(layer_name + '/pre_activations', preactivate)

        # Combine the feature maps if this is the last deconvolution
        if output_shape[-1] == 1:
            activations = act(tf.reduce_mean(preactivate,3,keep_dims=True), 'activation')
        else:
            activations = act(preactivate,'activation')

        tf.histogram_summary(layer_name + '/activations', activations)

        print layer_name + ' Shape: ', weights.get_shape(), ' with bias ', biases.get_shape(), ' padding', padding
        shape = activations.get_shape()
        print ' output : ', shape

        return activations, weights

def cnn_layer(input_tensor, convolution_shape, padding, layer_name, config, act=tf.nn.relu,strides=[1,1,1,1]):
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            name = layer_name + '/weights'
            weights = weight_variable(convolution_shape, name, config)
            variable_summaries(weights, name)
        with tf.name_scope('biases'):
            biases = bias_variable([convolution_shape[-1]], config)
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('convolution'):
            preactivate = conv2d(input_tensor, weights, padding=padding,strides=strides) + biases
            tf.histogram_summary(layer_name + '/pre_activations', preactivate)
        activations = act(preactivate, 'activation')
        tf.histogram_summary(layer_name + '/activations', activations)

        print layer_name + ' Shape: ', weights.get_shape(), ' with bias ', biases.get_shape(), ' padding', padding
        shape = activations.get_shape()
        print ' output : ', shape

        # channels = int(convolution_shape[-1])
        # img_size = int(shape[-2])
        # ix = img_size
        # iy = img_size
        # print channels, img_size
        #
        # cx = 0
        # cy = 0
        # if channels == 32:
        #     cx = 8
        #     cy = 4
        # if channels == 64:
        #     cx = 8
        #     cy = 8
        # if channels == 128:
        #     cx = 16
        #     cy = 8
        #
        # assert channels == cx * cy
        # assert cx != 0 and cy != 0
        #
        # if False:
        #     ## Prepare for visualization
        #     # Take only convolutions of first image, discard convolutions for other images.
        #     V = tf.slice(activations, (0, 0, 0, 0), (1, -1, -1, -1), name='slice_' + layer_name)
        #     V = tf.reshape(V, (img_size, img_size, channels))
        #
        #     # Reorder so the channels are in the first dimension, x and y follow.
        #     V = tf.transpose(V, (2, 0, 1))
        #     # Bring into shape expected by image_summary
        #     V = tf.reshape(V, (-1, img_size, img_size, 1))
        #     tf.image_summary('a_' + layer_name, V)
        #
        # if False:
        #     V = tf.slice(activations, (0, 0, 0, 0), (1, -1, -1, -1))
        #     V = tf.reshape(V, (iy, ix, channels))
        #
        #     ix = ix + 4
        #     iy = iy + 4
        #
        #     V = tf.image.resize_image_with_crop_or_pad(V, iy, ix)
        #
        #     V = tf.reshape(V, (iy, ix, cy, cx))
        #     V = tf.transpose(V, (2, 0, 3, 1))  # cy,iy,cx,ix
        #     # V = np.einsum('yxYX->YyXx',V)
        #     # image_summary needs 4d input
        #     V = tf.reshape(V, [1, cy * iy, cx * ix, 1])
        #     tf.image_summary('b_' + layer_name, V)

        # if layer_name == 'Convolution_1':
        #     grid = put_kernels_on_grid(weights, (8, 8),pad=2)
        #     tf.image_summary('c_'+ layer_name, grid)
        # if layer_name == 'Convolution_2':
        #     grid = put_kernels_on_grid(weights, (8, 8),pad=1)
        #     tf.image_summary(layer_name + '/features', grid)
        # if layer_name == 'Convolution_3':
        #     grid = put_kernels_on_grid(weights, (8, 16),pad=1)
        #     tf.image_summary(layer_name + '/features', grid)

        return activations, weights


def example_acnn(config):
    pass


def cnn(config):
    """

    :param config:
    :return:
    """
    batch_size = config['batch_size']
    input_dim = config['data']['image_shape']
    output_dim = config['data']['label_size']
    use_autoencoder = config['use_autoencoder']

    # Switch of the autoencoder if we don't know the batch size
    if batch_size == None:
        use_autoencoder = False

    if type(input_dim) == type([]):
        shape_1 = [batch_size, input_dim[0], input_dim[1]]
        shape_2 = [batch_size, input_dim[0], input_dim[1], 1]
    else:
        shape_1 = [batch_size, input_dim]
        shape_2 = [batch_size, int(np.sqrt(input_dim)), int(np.sqrt(input_dim)), 1]

    x = tf.placeholder(np.float32, shape=shape_1, name='Images')
    x_ = tf.reshape(x, shape=shape_2)

    if config['binary_softmax']:
        g = 2
    else:
        g = 1

    y_ = tf.placeholder(np.float32, shape=[batch_size, output_dim * g], name='Labels')

    print 'Input: ', x_.get_shape()
    print 'Outut: ', y_.get_shape()

    def flatten(network1, network2):
        with tf.name_scope('flatten'):
            prev_shape = ll(network1).get_shape()
            conv_output_size = int(1)
            for i in xrange(1, len(prev_shape)):
                conv_output_size *= int(prev_shape[i])
            flat = tf.reshape(ll(network1), [batch_size, conv_output_size])
            print ' flatten to ', flat.get_shape()
            network2.append((flat, None))

    def unflatten(network, decoder):
        with tf.name_scope('unflatten'):
            prev_shape = ll(network).get_shape()
            s = int(1)
            for i in xrange(1, len(prev_shape)):
                s *= int(prev_shape[i])
            unflatten = tf.reshape(ll(network), [batch_size, int(np.sqrt(s)), int(np.sqrt(s)), 1])
            print ' unflatten to ', unflatten.get_shape()
            decoder.append((unflatten, None))

    def leaky_relu(x, name):
        return tf.maximum(0.01 * x, x, name)

    def ll(network,i=-1):
        return network[i][0]

    print '\nENCODER'
    using_convolutions = True
    network = [(x_,None)]
    # http://stats.stackexchange.com/questions/65877/convergence-of-neural-network-weights
    #2209
    if config['network'] == 'gudi_test_network_0':
        using_convolutions = False
        print 'This network has no convolutions.'
    elif config['network'] == 'gudi_test_network_1':
        network.append(cnn_layer(ll(network), [5, 5, 1, 64], 'VALID', 'Convolution_1', config, act=leaky_relu))
        flatten(network)
    elif config['network'] == 'gudi_test_network_2':
        network.append(cnn_layer(ll(network), [5, 5, 1, 64], 'VALID', 'Convolution_1', config, act=leaky_relu))
        network.append(pool_layer('max',ll(network), ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',layer_name='Max_Pool_1'))
        flatten(network,network)
    elif config['network'] == 'gudi_test_network_3':
        network.append(cnn_layer(ll(network), [5, 5, 1, 64], 'VALID', 'Convolution_1', config, act=leaky_relu))
        network.append(pool_layer('max',ll(network), ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',layer_name='Max_Pool_1'))
        network.append(cnn_layer(ll(network), [5, 5, 64, 64], 'VALID', 'Convolution_2', config, act=leaky_relu))
        flatten(network, network)
    elif config['network'] == 'gudi_test_network_4':
        network.append(cnn_layer(ll(network), [5, 5, 1, 64], 'VALID', 'Convolution_1', config, act=leaky_relu))
        network.append(pool_layer('max',ll(network), ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',layer_name='Max_Pool_1'))
        network.append(cnn_layer(ll(network), [5, 5, 64, 64], 'VALID', 'Convolution_2', config, act=leaky_relu))
        network.append(cnn_layer(ll(network), [4, 4, 64, 128], 'VALID', 'Convolution_3', config, act=leaky_relu))
        flatten(network, network)
    elif config['network'] == 'gudi_test_network_4_avgpool':
        network.append(cnn_layer(ll(network), [5, 5, 1, 64], 'VALID', 'Convolution_1', config, act=leaky_relu))
        network.append(pool_layer('avg',ll(network), ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',layer_name='Max_Pool_1'))
        network.append(cnn_layer(ll(network), [5, 5, 64, 64], 'VALID', 'Convolution_2', config, act=leaky_relu))
        network.append(cnn_layer(ll(network), [4, 4, 64, 128], 'VALID', 'Convolution_3', config, act=leaky_relu))
        flatten(network, network)
    elif config['network'] == 'gudi_test_network_5':
        network.append(cnn_layer(ll(network), [5, 5, 1, 64], 'VALID', 'Convolution_1', config, act=leaky_relu))
        # network.append(pool_layer('max',ll(network), ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',layer_name='Max_Pool_1'))
        network.append(cnn_layer(ll(network), [5, 5, 64, 64], 'VALID', 'Convolution_2', config, act=leaky_relu,strides=[1,3,3,1]))
        network.append(cnn_layer(ll(network), [4, 4, 64, 128], 'VALID', 'Convolution_3', config, act=leaky_relu))
        flatten(network, network)
    elif config['network'] == 'gudi_test_network_6':
        network.append(cnn_layer(ll(network), [5, 5, 1, 64], 'VALID', 'Convolution_1', config, act=leaky_relu))
        # network.append(pool_layer('max',ll(network), ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',layer_name='Max_Pool_1'))
        network.append(cnn_layer(ll(network), [5, 5, 64, 64], 'VALID', 'Convolution_2', config, act=leaky_relu,strides=[1,3,3,1]))
        flatten(network, network)
    elif config['network'] == 'fc_1':
        flatten(network, network)
        network.append(nn_layer(ll(network), int(ll(network).get_shape()[1]), 2000, 'enc_fc_1',config,  act=leaky_relu))
        network.append(nn_layer(ll(network), int(ll(network).get_shape()[1]), 1500, 'enc_fc_2', config, act=leaky_relu))
        network.append(nn_layer(ll(network), int(ll(network).get_shape()[1]), 1000, 'enc_fc_3', config, act=leaky_relu))
        network.append(nn_layer(ll(network), int(ll(network).get_shape()[1]), 500, 'enc_fc_4', config, act=leaky_relu))
        network.append(nn_layer(ll(network), int(ll(network).get_shape()[1]), 100, 'enc_fc_5', config, act=leaky_relu))
    elif config['network'] == 'test':
        flatten(network, network)
        network.append(nn_layer(ll(network), int(ll(network).get_shape()[1]), 2, 'encoder',config,  act=leaky_relu))


    # Balance parameter
    alpha = tf.placeholder(np.float32)
    decoder = []


    if config['fc1_neuron_count'] > 0:
        with tf.name_scope('fully_connected_1'):
            network.append(nn_layer(ll(network), int(ll(network).get_shape()[1]), config['fc1_neuron_count'], 'fully_connected_1', config, act=leaky_relu))


    if use_autoencoder:
        print '\nDECODER:'
        s = 1
        for i in xrange(1, len(shape_2)):
            s *= int(shape_2[i])

        if config['autoencoder']['decoder'] == 'auto_cnn_1':
            print 'Resize:', [batch_size,22,22,64], '-->',[batch_size,43,43,64]
            decoder.append((tf.image.resize_images(ll(network,-2), 43, 43),None))
            decoder.append(dcnn_layer(ll(decoder), [5, 5, 1, 64], x_.get_shape(), 'VALID', 'Deconvolution_1', config))
        elif config['autoencoder']['decoder'] == 'auto_cnn_2':
            print 'Reshape:', ll(network).get_shape(), '-->',
            decoder.append( (tf.reshape(ll(network), shape=[batch_size,7,7,64]),None) )
            print ll(decoder).get_shape()

            print 'Resize:', ll(decoder).get_shape(), '-->',
            decoder.append( (tf.image.resize_nearest_neighbor(ll(decoder), size=[22,22]), None) )
            print ll(decoder).get_shape()

            decoder.append( dcnn_layer(ll(decoder), [5, 5, 1, 64], x_.get_shape(), 'VALID', 'Deconvolution_1', config,strides=[1,2,2,1]) )
        elif config['autoencoder']['decoder'] == 'auto_cnn_3':
            print 'Resize:', [batch_size,22,22,64], '-->',[batch_size,43,43,64]
            decoder.append((tf.image.resize_images(ll(network,-3), 43, 43),None))
            decoder.append(dcnn_layer(ll(decoder), [5, 5, 1, 64], x_.get_shape(), 'VALID', 'Deconvolution_1', config))
        elif config['autoencoder']['decoder'] == 'auto_cnn_4':
            decoder.append(dcnn_layer(ll(network,-2), [5, 5, 64, 64], [100,22,22,64], 'VALID', 'Deconvolution_1', config))
            print 'Resize:', ll(decoder).get_shape(), '-->', [100, 43, 43, 64]
            decoder.append((tf.image.resize_nearest_neighbor(ll(decoder), size=[43, 43]),None))
            decoder.append(dcnn_layer(ll(decoder), [5, 5, 1, 64], x_.get_shape(), 'VALID', 'Deconvolution_2', config))
        elif config['autoencoder']['decoder'] == 'auto_cnn_6':
            decoder.append(dcnn_layer(ll(network,-3), [5, 5, 64, 64], [100,22,22,64], 'VALID', 'Deconvolution_1', config))
            print 'Resize:', ll(decoder).get_shape(), '-->', [100, 43, 43, 64]
            decoder.append((tf.image.resize_nearest_neighbor(ll(decoder), size=[43, 43]),None))
            decoder.append(dcnn_layer(ll(decoder), [5, 5, 1, 64], x_.get_shape(), 'VALID', 'Deconvolution_2', config))
        elif config['autoencoder']['decoder'] == 'auto_fc_1':
            decoder.append(nn_layer(ll(network), int(ll(network).get_shape()[1]), 1000, 'dec_fc_2', config, act=leaky_relu))
            decoder.append(nn_layer(ll(decoder), int(ll(decoder).get_shape()[1]), 1500, 'dec_fc_3', config, act=leaky_relu))
            decoder.append(nn_layer(ll(decoder), int(ll(decoder).get_shape()[1]), 2000, 'dec_fc_4', config, act=leaky_relu))
            decoder.append(nn_layer(ll(decoder), int(ll(decoder).get_shape()[1]), 2209, 'dec_fc_5', config, act=leaky_relu))
        elif config['autoencoder']['decoder'] == 'test':
            decoder.append(nn_layer(ll(network), int(ll(network).get_shape()[1]), 2209, 'decoder', config, act=leaky_relu))



        # elif config['autoencoder']['decoder'] == 'auto_cnn_1':
        #     decoder.append(nn_layer(ll(network), int(ll(network).get_shape()[1]),s, 'decoder_fully_connected_1', config, act=leaky_relu))
        # elif config['autoencoder']['decoder'] == 'auto_fc_2':
        #     decoder.append(nn_layer(ll(network), int(ll(network).get_shape()[1]), 1000, 'decoder_fully_connected_1', config,act=leaky_relu))
        #     decoder.append(nn_layer(ll(decoder), int(ll(decoder).get_shape()[1]), s, 'decoder_fully_connected_2', config,act=leaky_relu))
        # elif config['autoencoder']['decoder'] == 'gudi_test_network_1':
        #     unflatten(network,decoder)
        #     # ll(network,2) means start the decoder from the second to last bit of the autoencoder
        #     decoder.append(dcnn_layer(ll(decoder), [5, 5, 1, 64], x_.get_shape(), 'VALID', 'Deconvolution_1', config))
        # elif config['autoencoder']['decoder'] == 'gudi_test_network_2':
        #     unflatten(network, decoder)
        #     decoder.append(dcnn_layer(ll(decoder,-2), [5, 5, 1, 64], x_.get_shape(), 'VALID', 'Deconvolution_1', config))
        # elif config['autoencoder']['decoder'] == 'gudi_test_network_3':
        #     unflatten(network, decoder)
        #     decoder.append(dcnn_layer(ll(decoder), [5, 5, 64, 64], [batch_size,22,22,64], 'VALID', 'Deconvolution_1', config))
        #     print 'Resize:', [batch_size,22,22,64], '-->',[batch_size,43,43,64]
        #     decoder.append((tf.image.resize_images(ll(decoder), 43, 43),None))
        #     decoder.append(dcnn_layer(ll(decoder), [5, 5, 1, 64], x_.get_shape(), 'VALID', 'Deconvolution_2', config))

        y_auto = ll(decoder)
        y_image = tf.reshape(y_auto, shape=shape_2)
        auto_loss = tf.sqrt(tf.reduce_mean(tf.square(x_ - y_image)))

    print '\nCLASSIFER:'

    if config['fc2_neuron_count'] > 0:
        with tf.name_scope('fully_connected_2'):
            network.append(nn_layer(ll(network), int(ll(network).get_shape()[1]), config['fc2_neuron_count'], 'fully_connected_2', config, act=leaky_relu))


    if config['fc3_neuron_count'] > 0:
        with tf.name_scope('fully_connected_3'):
            network.append(nn_layer(ll(network), int(ll(network).get_shape()[1]), config['fc3_neuron_count'], 'fully_connected_3', config, act=leaky_relu))

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        network.append((tf.nn.dropout(ll(network), keep_prob,seed=config['seed']),None))

    if config['final_activation'] == 'softmax':
        final_act = tf.nn.softmax
    else:
        final_act = tf.nn.relu

    if config['binary_softmax']:
        cost_outputs = []
        metric_outputs = []
        for i in xrange(output_dim):
            l, _ = nn_layer(ll(network), int(ll(network).get_shape()[1]), 2, 'output_' + str(i), config, act=final_act)
            cost_outputs.append(l)
            metric_outputs.append(tf.slice(l, [0, 0], [-1, 1]))

        with tf.name_scope('training'):
            y_train = tf.concat(concat_dim=1, values=cost_outputs)
        with tf.name_scope('inference'):
            y_conv = tf.concat(concat_dim=1, values=metric_outputs)
    else:
        y_conv = nn_layer(ll(network), int(ll(network).get_shape()[1]), output_dim, 'output', config, act=final_act)
        y_train = y_conv

    # if config['ignore_empty_labels']:
    #     with tf.name_scope('sparse_weights'):
    #         y_conv = tf.transpose(tf.mul(tf.reduce_sum(tf.cast(y_,tf.float32),1),tf.transpose(y_conv_unweighted)))
    # else:
    #     y_conv = y_conv_unweighted

    print 'Output shape : ', y_conv.get_shape()

    THRESHOLD = config['threshold']
    """
    Accuracy dodgyness
    """
    with tf.name_scope('accuracy_madness'):
        # a = tf.cast(tf.greater(y_conv,THRESHOLD),tf.float32)
        # b = tf.cast(tf.greater(y_,THRESHOLD),tf.float32)
        # c = tf.abs(a-b)
        # ab_sum = tf.reduce_sum(a,1)+tf.reduce_sum(b,1)
        # accuracy = tf.reduce_mean(tf.Variable(1.0)-tf.reduce_sum(c,1)/ab_sum)
        accuracy = tf.Variable(1.0)

    with tf.name_scope('train'):
        with tf.name_scope('lmsq_loss'):
            classifer_lmsq_loss = tf.sqrt(tf.reduce_mean(tf.square(y_train - y_)))
            tf.scalar_summary('loss/lmsq_loss', classifer_lmsq_loss)
        with tf.name_scope('cross_entropy'):
            if batch_size != None:
                norm = float(output_dim * batch_size)
            else:
                norm = float(output_dim)
            cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_train, 1e-10, 1.0)))/norm
            tf.scalar_summary('loss/cross entropy', cross_entropy)

        r = config['learning_rate']
        if config['cost_function'] == 'cross_entropy':
            loss = cross_entropy
        else:
            loss = classifer_lmsq_loss

        if use_autoencoder:
            combined_loss = (1.0-alpha)*loss + alpha*auto_loss
        else:
            combined_loss = loss

        if config['seed_randomness']:
            tf.set_random_seed(config['seed'])
        if config['optimizer'] == 'adam':
            train_step = tf.train.AdamOptimizer(r).minimize(combined_loss)
        else:
            train_step = tf.train.GradientDescentOptimizer(r).minimize(combined_loss)
    del network
    return {'x' : x,
            'y' : y_,
            'train_step' : train_step,
            'loss' : loss,
            'y_conv' : y_conv,
            'y_auto': y_auto,
            'y_image' : y_image,
            'output_dim' : output_dim,
            'keep_prob' : keep_prob,
            'classifer_lmsq_loss' : classifer_lmsq_loss,
            'cross_entropy' : cross_entropy,
            'accuracy' : accuracy,
            'alpha' : alpha,
            'auto_loss' : auto_loss,
            'batch_size' : batch_size}
