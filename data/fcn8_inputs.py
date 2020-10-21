import tensorflow as tf
import os

VGG_MEAN = [103.939, 116.779, 123.68]
IK_MEAN = [147.97, 143.40, 133.01, 177.56]

def _decode_img(feature, shape, dtype):
    image = tf.decode_raw(feature, dtype)
    image_shape = tf.stack(shape)

    return tf.reshape(image, image_shape)

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