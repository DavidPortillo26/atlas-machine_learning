#!/usr/bin/env python3
"""
    Tensorflow project
"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """ Calculates softmax cross-entropy loss of a prediction
        y is placeholder for labels of input data
        y_pred is tensor containing networks predictions
    """
    m = tf.losses.softmax_cross_entropy(
        y,
        y_pred)
    return m
