#!/usr/bin/env python3
"""
    Tensorflow project
"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """ Creates training operation for NN
    """
    grady = tf.train.GradientDescentOptimizer(alpha, name='GradientDescent')
    '''
    tf.contrib.training.create_train_op(
        loss,
        opti)'''
    return grady.minimize(loss)
