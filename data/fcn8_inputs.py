from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import os
from scipy import stats
import numpy as np

VGG_MEAN = [103.939, 116.779, 123.68]
IK_MEAN = [147.97, 143.40, 133.01, 177.56]

def _decode_img(feature, shape, dtype):
    image = tf.decode_raw(feature, dtype)
    image_shape = tf.stack(shape)

    return tf.reshape(image, image_shape)

def _parse_ik2tcc(example_proto):
    '''Parses a TFRecord from file to examples'''
    features = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string),
    }

    parsed_features = tf.parse_single_example(example_proto, features)

    height = tf.cast(parsed_features['height'], tf.int32)
    width = tf.cast(parsed_features['width'], tf.int32)
    depth = tf.cast(parsed_features['depth'], tf.int32)

    image = _decode_img(
        parsed_features['image'],
        [depth, height, width],
        tf.float64
    )

    image = tf.transpose(image[:3], perm=[1, 2, 0])

    # scale 0 to 255 for compatibility
    max_by_band = tf.reduce_max(image, axis=[0, 1])
    image = (image / max_by_band) * 255
    zeros = tf.equal(image, tf.constant(0., dtype=tf.float64))
    # normalisation
    image = image - VGG_MEAN
    image = tf.where(zeros, tf.zeros_like(image), image)
    label = _decode_img(
        parsed_features['label'],
        [height, width],
        tf.float64
    )

    return image, label

def _parse_ik_full(example_proto):
    '''Parses a TFRecord from file to examples'''
    features = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string),
    }

    parsed_features = tf.parse_single_example(example_proto, features)

    height = tf.cast(parsed_features['height'], tf.int32)
    width = tf.cast(parsed_features['width'], tf.int32)
    depth = tf.cast(parsed_features['depth'], tf.int32)

    image = _decode_img(
        parsed_features['image'],
        [depth, height, width],
        tf.float64
    )

    image = tf.transpose(image, perm=[1, 2, 0])

    # scale 0 to 255 for compatibility
    max_by_band = tf.reduce_max(image, axis=[0, 1])
    image = (image / max_by_band) * 255
    zeros = tf.equal(image, tf.constant(0., dtype=tf.float64))
    # normalisation
    image = image - IK_MEAN
    image = tf.where(zeros, tf.zeros_like(image), image)
    label = _decode_img(
        parsed_features['label'],
        [height, width],
        tf.float64
    )

    return image, label

def _parse_ik(example_proto, n_layers, norm_vgg=False):
    '''Parses a TFRecord from file to examples'''
    features = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string),
    }

    parsed_features = tf.parse_single_example(example_proto, features)

    height = tf.cast(parsed_features['height'], tf.int32)
    width = tf.cast(parsed_features['width'], tf.int32)
    depth = tf.cast(parsed_features['depth'], tf.int32)

    image = _decode_img(
        parsed_features['image'],
        [depth, height, width],
        tf.float64
    )

    image = tf.transpose(image[:n_layers], perm=[1, 2, 0])
    # scale 0 to 255 for compatibility
    max_by_band = tf.reduce_max(image, axis=[0, 1])
    image = (image / max_by_band) * 255
    zeros = tf.equal(image, tf.constant(0., dtype=tf.float64))
    # normalisation
    if norm_vgg:
        image = image - VGG_MEAN[:n_layers]
    else:
        image = image - IK_MEAN[:n_layers]

    image = tf.where(zeros, tf.zeros_like(image), image)
    label = _decode_img(
        parsed_features['label'],
        [height, width],
        tf.float64
    )

    return image, label

def _generate_batch(tfrecords, batch_size, in_layers, shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecords)
    dataset = dataset.map(lambda x: _parse_ik(x, in_layers))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=200)

    dataset = dataset.batch(batch_size)

    return dataset

def _generate_image_label_batch(tfrecords, batch_size, shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecords)
    dataset = dataset.map(map_func=_parse_ik2tcc)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=200)

    dataset = dataset.batch(batch_size)

    return dataset

def _generate_image_label_batch_full(tfrecords, batch_size, shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecords)
    dataset = dataset.map(map_func=_parse_ik_full)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=200)

    dataset = dataset.batch(batch_size)

    return dataset

def validate2008(batch_size):
    records = [
        '/home/dan/projects/deepjet/data/samples08/held2008val.tfrecords'
    ]
    return get_records(records, batch_size)

def get_record_batches(records, batch_size, in_layers, shuffle=True):
    for record in records:
        if not os.path.exists(record):
            print('File {} not found'.format(record))
            return None

    batch_gen = _generate_batch(
        records,
        batch_size,
        in_layers,
        shuffle=shuffle
    )
    return batch_gen


def get_records_full(records, batch_size):
    for record in records:
        if not os.path.exists(record):
            print('File {} not found'.format(record))
            return None

    return _generate_image_label_batch_full(records, batch_size,
                                            shuffle=True)

def get_records(records, batch_size):
    for record in records:
        if not os.path.exists(record):
            print('File {} not found'.format(record))
            return None

    return _generate_image_label_batch(records, batch_size,
                                            shuffle=True)

def input2008(batch_size):
    records = [
        '/home/dan/projects/deepjet/data/samples08/held2008train.tfrecords'
    ]
    return get_records(records, batch_size)

def scale_cdf_clip(pixels, min=[], max=[]):
    out = np.zeros_like(pixels)
    if not min:
        min = np.ones(len(pixels))*.025
    if not max:
        max = np.ones(len(pixels))*.99

    for b in range(len(pixels)):
        band = pixels[b]

        minv, maxv = cdf_clip(band[band>0], min=min[b], max=max[b])

        out[b] = (band - minv) / (maxv - minv)
    out[pixels==0] = 0.
    return out

def cdf_clip(data, min=.025, max=.99, nbins=100):
    res = stats.cumfreq(data, numbins=nbins)
    x = res.lowerlimit + np.linspace(
     0,
     res.binsize*res.cumcount.size,
     res.cumcount.size
    )

    nres = res.cumcount / res.cumcount.max()

    a = x[np.abs(nres - min).argmin()]
    b = x[np.abs(nres - max).argmin()]
    c = a - .1 * (b-a)
    d = b + .5 * (b-a)

    return c, d
