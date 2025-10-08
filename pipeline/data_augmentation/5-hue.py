#!/usr/bin/env python3
"""
Module that changes the hue of an
image using TensorFlow.
"""

import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image.

    Args:
        image: A 3D tf.Tensor containing the image to change.
        delta: The amount the hue should change (float).

    Returns:
        The hue-altered image (tf.Tensor).
    """
    return tf.image.adjust_hue(image, delta)
