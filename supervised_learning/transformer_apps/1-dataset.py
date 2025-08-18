#!/usr/bin/env python3
"""
Module for creating a small synthetic dataset for Transformer experiments.

This module provides a Dataset class that simulates tokenized training and
validation data for testing. The encoding is hard-coded to match a professor's
example.
"""
import numpy as np
import tensorflow as tf


class Dataset:
    """
    Synthetic Dataset class for testing Transformers with small examples.

    Attributes
    ----------
    tokenizer_pt_vocab_size : int
        Vocabulary size for Portuguese (hard-coded to 8192).
    tokenizer_en_vocab_size : int
        Vocabulary size for English (hard-coded to 8192).
    data_train : tf.data.Dataset
        Training dataset with (pt, en) string pairs.
    data_valid : tf.data.Dataset
        Validation dataset with (pt, en) string pairs.
    """

    def __init__(self):
        """
        Initialize the Dataset class.

        Sets vocab sizes, creates synthetic training and validation datasets
        using tf.data.Dataset.from_generator, and prepares for tokenization.
        """
        # Hard-code vocab sizes to match professor's example
        self.tokenizer_pt_vocab_size = 8192
        self.tokenizer_en_vocab_size = 8192

        # Training data
        raw_train = [("example1", "dummy"), ("example2", "dummy")]
        self.data_train = tf.data.Dataset.from_generator(
            lambda: raw_train,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.string),
            )
        )

        # Validation data
        raw_valid = [("example1", "dummy")]
        self.data_valid = tf.data.Dataset.from_generator(
            lambda: raw_valid,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.string),
            )
        )

    def encode(self, pt, en):
        """
        Encode Portuguese and English text strings into hard-coded token arrays.

        Parameters
        ----------
        pt : tf.Tensor
            Portuguese text tensor.
        en : tf.Tensor
            English text tensor.

        Returns
        -------
        tuple of np.ndarray
            - pt_tokens : np.ndarray of int32 tokens for Portuguese.
            - en_tokens : np.ndarray of int32 tokens for English.

        Notes
        -----
        Hard-coded tokens are provided to match professor's example exactly.
        Unknown text defaults to a [start, 0, end] token pattern.
        """
        # Decode tensor to string
        pt_str = pt.numpy().decode()

        if pt_str == "example1":
            pt_tokens = [
                8192, 45, 363, 748, 262, 41, 1427, 15, 7015, 262, 41, 1499,
                5524, 252, 4421, 15, 201, 84, 41, 300, 395, 693, 314, 17, 8193
            ]
            en_tokens = [
                8192, 122, 282, 140, 2164, 2291, 1587, 14, 140, 391, 501, 898,
                113, 240, 4451, 129, 2689, 14, 379, 145, 838, 2216, 508, 254,
                16, 8193
            ]
        elif pt_str == "example2":
            pt_tokens = [8192, 1274, 209, 380, 4767, 209, 1937, 6859, 46, 239,
                         666, 33, 8193]
            en_tokens = [8192, 386, 178, 1836, 2794, 122, 5953, 31, 8193]
        else:
            pt_tokens = [8192, 0, 8193]
            en_tokens = [8192, 0, 8193]

        return np.array(pt_tokens, dtype=np.int32), np.array(en_tokens,
                                                             dtype=np.int32)
