#!/usr/bin/env python3
"""Script to create a layer"""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    method to create a TF layer
    Args:
        prev: tensor of the previous layer
        n: n nodes created
        activation: activation function

    Returns: Layer created with shape n

    """
    # Average number of inputs and output connections.
    layer = tf.layers.Dense(
        activation=activation,
        name="layer",
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg')
    )
    return layer(prev)
