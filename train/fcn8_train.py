from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf

def _expand_label(label, classes):
    # element wise expansion of labels
    label_expand = list(map(lambda x: tf.equal(label, x), classes))
    stack = tf.stack(label_expand, axis=2)

    return tf.to_float(stack)

def batch_expand_labels(labels, classes):
    batch_labels = tf.map_fn(
        lambda x: _expand_label(x, classes),
        labels,
        dtype=tf.float32
    )
    return batch_labels

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels,
        logits=logits,
        name='cross_entropy_per_example'
    )
    mean_xentropy = tf.reduce_mean(xentropy, name='cross_entropy')
    tf.add_to_collection('losses', mean_xentropy)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return total_loss

def pixel_accuracy(labels, pred):
    """Total pixel accuarcy (user)"""
    ac = tf.divide(
        tf.reduce_sum(tf.cast(tf.equal(pred, labels), tf.float32)),
        tf.cast(tf.size(labels), tf.float32)
    )
    return ac
