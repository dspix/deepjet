from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
import numpy as np

NUM_OF_CLASSESS = 5

def _bilinear_filter(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        centre = factor - 1
    else:
        centre = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - centre) / factor) * \
           (1 - abs(og[1] - centre) / factor)

def _upscore_filter(shape):
    h, w, nin, nout = shape
    filt = _bilinear_filter(w) #h, w for square
    weights = np.zeros(shape)
    for l in range(nin):
        weights[:, :, l, l] = filt
    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    up_filt =  tf.get_variable(
        name='upscore_filter', initializer=init,
        shape=weights.shape)

    return up_filt

def _conv_relu(name, bottom, nout, ks=[1, 1], strides=[1, 1, 1, 1],
               in_layers=None, trainable=True, return_relu=True):
    """
    """
    with tf.variable_scope(name) as scope:
        if in_layers is None:
            nin = bottom.shape.as_list()[3]
        else:
            nin = in_layers
        shape = ks + [nin, nout]
        kernel = tf.get_variable(
            'weights', shape,
            dtype=tf.float32, trainable=trainable)
        conv = tf.nn.conv2d(
            bottom, kernel, strides, padding='SAME',
            name=name)
        biases = tf.get_variable('biases', [nout], trainable=trainable)
        pre_activation = tf.nn.bias_add(
            conv, biases,
            name=name)
        if return_relu:
            relu = tf.nn.relu(
                pre_activation, name=name.replace('conv', 'relu'))

            return pre_activation, relu

        return pre_activation

def _max_pool(name, bottom):
    maxp = tf.nn.max_pool(
        bottom, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME', name=name)

    return maxp

def _upscale(name, bottom, nin, nout, out_shape, ks=[1, 1],
             strides=[1, 2, 2, 1], trainable=True):
    with tf.variable_scope(name) as scope:
        shape = ks + [nin, nout]
        kernal = _upscore_filter(shape)
        conv = tf.nn.conv2d_transpose(
            bottom, kernal,
            out_shape, strides,
            padding='SAME', name=name)
        return conv

def inference(x, in_layers=3):
    # convolution layers
    conv1_1, relu1_1 = _conv_relu('conv1_1', x, 64, ks=[3, 3],
                                  in_layers=in_layers)
    conv1_2, relu1_2 = _conv_relu('conv1_2', relu1_1, 64, ks=[3, 3])
    pool1 = _max_pool('pool1', relu1_2)

    conv2_1, relu2_1 = _conv_relu('conv2_1', pool1, 128, ks=[3, 3])
    conv2_2, relu2_2 = _conv_relu('conv2_2', relu2_1, 128, ks=[3, 3])
    pool2 = _max_pool('pool2', relu2_2)

    conv3_1, relu3_1 = _conv_relu('conv3_1', pool2, 256, ks=[3, 3])
    conv3_2, relu3_2 = _conv_relu('conv3_2', relu3_1, 256, ks=[3, 3])
    conv3_3, relu3_3 = _conv_relu('conv3_3', relu3_2, 256, ks=[3, 3])
    pool3 = _max_pool('pool3', relu3_3)

    conv4_1, relu4_1 = _conv_relu('conv4_1', pool3, 512, ks=[3, 3])
    conv4_2, relu4_2 = _conv_relu('conv4_2', relu4_1, 512, ks=[3, 3])
    conv4_3, relu4_3 = _conv_relu('conv4_3', relu4_2, 512, ks=[3, 3])
    pool4 = _max_pool('pool4', relu4_3)

    conv5_1, relu5_1 = _conv_relu('conv5_1', pool4, 512, ks=[3, 3])
    conv5_2, relu5_2 = _conv_relu('conv5_2', relu5_1, 512, ks=[3, 3])
    conv5_3, relu5_3 = _conv_relu('conv5_3', relu5_2, 512, ks=[3, 3])
    pool5 = _max_pool('pool5', relu5_3)

    # Fully convolutional with dropout
    fc6, relu6 = _conv_relu('fc6', pool5, 4096, ks=[7, 7])
    drop6 = tf.nn.dropout(relu6, keep_prob=.5)
    fc7, relu7 = _conv_relu('fc7', drop6, 4096, ks=[1, 1])
    drop7 = tf.nn.dropout(relu7, keep_prob=.5)

    # Skips
    fc7_score = _conv_relu(
        '7_score', drop7,
        NUM_OF_CLASSESS, return_relu=False)
    pool4_score = _conv_relu(
        'pool4_score', pool4,
        NUM_OF_CLASSESS, return_relu=False)
    pool4_nin = pool4_score.get_shape()[3].value
    fc7_upscore = _upscale(
        'upscale_pool4', fc7_score,
        pool4_nin, NUM_OF_CLASSESS,
        tf.shape(pool4_score), ks=[4, 4])
    fuse_pool4 = tf.add(pool4_score, fc7_upscore, name='fuse_pool4')

    pool3_score = _conv_relu(
        'pool3_score', pool3,
        NUM_OF_CLASSESS, return_relu=False)
    pool3_nin = pool3_score.get_shape()[3].value
    fuse_pool4_upscore = _upscale(
        'upscale_pool3', fuse_pool4,
        pool3_nin, NUM_OF_CLASSESS,
        tf.shape(pool3_score), ks=[4, 4])
    fuse_pool3 = tf.add(
        pool3_score, fuse_pool4_upscore,
        name='fuse_pool3')

    # Upscale to image size
    shape = tf.shape(x)
    out_shape = tf.stack([shape[0], shape[1], shape[2], 5])
    score = _upscale(
        'upscale_orig', fuse_pool3,
        pool3_nin, NUM_OF_CLASSESS,
        out_shape, ks=[16, 16], strides=[1, 8, 8, 1])

    return score

def train(total_loss, learning_rate):
    op = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads = op.compute_gradients(total_loss, var_list=None)
    return op.apply_gradients(grads)
