#!/usr/bin/env python3
"""Set up the RMSProp optimization algorithm in TensorFlow 2.x"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Creates a TensorFlow optimizer using the RMSProp optimization
    algorithm.

    Args:
        alpha (float): Learning rate for the optimizer, controlling
        the step size.
        beta2 (float): Discounting factor for the moving average of
        squared gradients (RMSProp weight).
        epsilon (float): Small constant to avoid division by zero
        during updates.

    Returns:
        tf.optimizers.Optimizer: An RMSProp optimizer instance
        configured with the provided parameters.
    """
    # Create the RMSProp optimizer using the provided parameters
    optimizer = tf.optimizers.RMSprop
    (learning_rate=alpha, rho=beta2, epsilon=epsilon)

    return optimizer
