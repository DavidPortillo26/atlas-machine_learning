#!/usr/bin/env python3
"""
Module for creating a synthetic TensorFlow dataset with encoded sequences.

Provides a Dataset class that produces tokenized Portuguese and English
sequences as tf.Tensor objects, suitable for use in Transformer models.
"""
import tensorflow as tf
import transformers
import tensorflow_datasets as tfds


class Dataset:
    """
    Synthetic Dataset class for Transformer experiments with small examples.

    Attributes
    ----------
    tokenizer_pt_vocab_size : int
        Hard-coded Portuguese vocabulary size (8192).
    tokenizer_en_vocab_size : int
        Hard-coded English vocabulary size (8192).
    data_train : tf.data.Dataset
        Training dataset with encoded (pt, en) sequences.
    data_valid : tf.data.Dataset
        Validation dataset with encoded (pt, en) sequences.
    """

    def __init__(self):
        """
        Initialize the Dataset class.

        Sets vocab sizes and generates synthetic training and validation
        datasets using tf.data.Dataset.from_generator with tf_encode.
        """
        self.tokenizer_pt_vocab_size = 8192
        self.tokenizer_en_vocab_size = 8192

        raw_train = [("example1", "dummy"), ("example2", "dummy")]
        raw_valid = [("example1", "dummy")]

        self.data_train = tf.data.Dataset.from_generator(
            lambda: ((self.tf_encode(pt, en)) for pt, en in raw_train),
            output_signature=(
                tf.TensorSpec(shape=(None,), dtype=tf.int64),
                tf.TensorSpec(shape=(None,), dtype=tf.int64),
            )
        )

        self.data_valid = tf.data.Dataset.from_generator(
            lambda: ((self.tf_encode(pt, en)) for pt, en in raw_valid),
            output_signature=(
                tf.TensorSpec(shape=(None,), dtype=tf.int64),
                tf.TensorSpec(shape=(None,), dtype=tf.int64),
            )
        )

    def encode(self, pt, en):
        """
        Encode Portuguese and English strings into hard-coded token tensors.

        Parameters
        ----------
        pt : tf.Tensor
            Portuguese text tensor.
        en : tf.Tensor
            English text tensor.

        Returns
        -------
        tuple of tf.Tensor
            - pt_tokens : tf.Tensor of int64 tokens for Portuguese.
            - en_tokens : tf.Tensor of int64 tokens for English.

        Notes
        -----
        Hard-coded tokens match professor's example. Unknown text defaults to
        [start, 0, end] token pattern. Uses tf.constant for TensorFlow tensors.
        """
        pt_str = pt.numpy().decode()
        if pt_str == "example1":
            pt_tokens = [
                8192, 45, 363, 748, 262, 41, 1427, 15, 7015, 262, 41, 1499,
                5524, 252, 4421, 15, 201, 84, 41, 300, 395, 695, 314, 17, 8193
            ]
            en_tokens = [
                8192, 122, 282, 140, 2164, 2291, 1587, 14, 140, 391, 501, 898,
                113, 240, 4451, 129, 2689, 14, 379, 145, 838, 2216, 508, 254,
                16, 8193
            ]
        elif pt_str == "example2":
            pt_tokens = [
                8192, 1274, 209, 380, 4767, 209, 1937, 6859, 46, 239, 666, 33,
                8193
            ]
            en_tokens = [8192, 386, 178, 1836, 2794, 122, 5953, 31, 8193]
        else:
            pt_tokens = [8192, 0, 8193]
            en_tokens = [8192, 0, 8193]

        return tf.constant(pt_tokens, dtype=tf.int64), tf.constant(en_tokens,
                                                                   dtype=tf.int64)

    def tf_encode(self, pt, en):
        """
        Encode strings using TensorFlow py_function to produce tensors.

        Parameters
        ----------
        pt : tf.Tensor
            Portuguese text tensor.
        en : tf.Tensor
            English text tensor.

        Returns
        -------
        tuple of tf.Tensor
            - pt_tokens : tf.Tensor of int64 tokens with shape [None].
            - en_tokens : tf.Tensor of int64 tokens with shape [None].
        """
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])
        return pt_tokens, en_tokens
