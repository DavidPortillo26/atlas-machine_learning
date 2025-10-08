#!/usr/bin/env python3
"""Module that flips an image horizontally using TensorFlow."""

import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally.

    Args:
        image: A 3D tf.Tensor representing the image to flip.

    Returns:
        The horizontally flipped image (tf.Tensor).
    """
    return tf.image.flip_left_right(image)
