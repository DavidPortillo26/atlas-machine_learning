#!/usr/bin/env python3
"""
    Tensorflow project
"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """ Calculates accuracy (cost) of a prediction """
    '''loss = tf.metrics.accuracy(
        y,
        y_pred, name="Meanie")
    print(loss)
    m = tf.convert_to_tensor(loss[0], name="Mean")'''
    m = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    return tf.reduce_mean(tf.cast(m, tf.float32), name="Mean")
