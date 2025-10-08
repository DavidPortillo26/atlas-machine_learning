#!/usr/bin/env python3
"""Module that rotates an image 90 degrees counter-clockwise using TensorFlow."""

import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise.

    Args:
        image: A 3D tf.Tensor representing the image to rotate.

    Returns:
        The rotated image (tf.Tensor).
    """
    return tf.image.rot90(image, k=1)
