#!/usr/bin/env python3
"""Module that performs a random crop on an image using TensorFlow."""
import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop on an image.

    Args:
        image: A 3D tf.Tensor representing the image to crop.
        size: A tuple (height, width, channels)
        representing the size of the crop.

    Returns:
        The randomly cropped image (tf.Tensor).
    """
    return tf.image.random_crop(image, size)
