#!/usr/bin/env python3
"""Set up gradient descent with momentum optimization algorithm in TensorFlow."""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Creates a TensorFlow optimizer using gradient descent with momentum.

    Args:
        alpha: Learning rate (scalar).
        beta1: Momentum weight (scalar).

    Returns:
        A TensorFlow optimizer object configured for momentum optimization.
    """
    # Create the optimizer using Stochastic Gradient Descent (SGD) with momentum
    optimizer = tf.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    
    return optimizer
