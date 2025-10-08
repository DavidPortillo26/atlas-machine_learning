#!/usr/bin/env python3
import tensorflow as tf
"""
Module that defines a function to flip images horizontally using TensorFlow.

This module contains the function `flip_image(image)`, which takes a 3D
TensorFlow tensor representing an image and returns a horizontally flipped
version of that image. It can be used for data augmentation or preprocessing
in computer vision tasks.
"""


def flip_image(image):
    """
    Flips an image horizontally.

    Args:
        image: A 3D tf.Tensor representing the image to flip.

    Returns:
        The horizontally flipped image (tf.Tensor).
    """
    return tf.image.flip_left_right(image)
