#!/usr/bin/env python3
"""
    Tensorflow project
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """ Returns two placeholders, x and y for NN """
    x = tf.placeholder("float", shape=(None, nx), name='x')
    y = tf.placeholder("float", shape=(None, classes), name='y')
    return x, y
