#!/usr/bin/env python3
"""
Module for loading, encoding, and batching the TED HRLR Portuguese-to-English
dataset for Transformer training.
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Dataset class for preparing TED HRLR Portuguese-to-English data.

    Attributes
    ----------
    tokenizer_pt_vocab_size : int
        Hard-coded Portuguese vocabulary size (8192).
    tokenizer_en_vocab_size : int
        Hard-coded English vocabulary size (8192).
    batch_size : int
        Number of examples per batch.
    max_len : int
        Maximum sequence length for input and target sequences.
    tokenizer_pt : tfds.deprecated.text.SubwordTextEncoder
        Subword tokenizer for Portuguese.
    tokenizer_en : tfds.deprecated.text.SubwordTextEncoder
        Subword tokenizer for English.
    data_train : tf.data.Dataset
        Preprocessed and batched training dataset.
    data_valid : tf.data.Dataset
        Preprocessed and batched validation dataset.
    """

    def __init__(self, batch_size, max_len):
        """
        Initialize the Dataset class.

        Loads the TED HRLR Portuguese-to-English dataset, builds subword
        tokenizers, and prepares batched and padded datasets for training and
        validation.

        Parameters
        ----------
        batch_size : int
            Number of examples per batch.
        max_len : int
            Maximum sequence length for input and target sequences.
        """
        self.tokenizer_pt_vocab_size = 8192
        self.tokenizer_en_vocab_size = 8192
        self.batch_size = batch_size
        self.max_len = max_len

        # Load TED HRLR dataset
        examples, metadata = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            as_supervised=True,
            with_info=True
        )
        train_examples, val_examples = examples['train'], examples['validation']

        # Build subword tokenizers from training corpus
        self.tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.decode("utf-8") for pt, en in tfds.as_numpy(train_examples)),
            target_vocab_size=self.tokenizer_pt_vocab_size
        )
        self.tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.decode("utf-8") for pt, en in tfds.as_numpy(train_examples)),
            target_vocab_size=self.tokenizer_en_vocab_size
        )

        # Encoder function to add start/end tokens and convert to tf.Tensor
        def encode(pt, en):
            pt_tokens = [self.tokenizer_pt.vocab_size] + \
                        self.tokenizer_pt.encode(pt.numpy().decode("utf-8")) + \
                        [self.tokenizer_pt.vocab_size + 1]
            en_tokens = [self.tokenizer_en.vocab_size] + \
                        self.tokenizer_en.encode(en.numpy().decode("utf-8")) + \
                        [self.tokenizer_en.vocab_size + 1]
            return tf.constant(pt_tokens, dtype=tf.int64), \
                   tf.constant(en_tokens, dtype=tf.int64)

        # TensorFlow wrapper for encoding
        def tf_encode(pt, en):
            pt_tokens, en_tokens = tf.py_function(
                encode, [pt, en], [tf.int64, tf.int64]
            )
            pt_tokens.set_shape([None])
            en_tokens.set_shape([None])
            return pt_tokens, en_tokens

        # Prepare training dataset: map, filter, shuffle, batch, prefetch
        self.data_train = (train_examples
                           .map(tf_encode,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
                           .filter(lambda pt, en: tf.logical_and(tf.size(pt) <= max_len,
                                                                 tf.size(en) <= max_len))
                           .cache()
                           .shuffle(20000)
                           .padded_batch(batch_size, padded_shapes=([None], [None]))
                           .prefetch(tf.data.experimental.AUTOTUNE))

        # Prepare validation dataset: map, filter, batch
        self.data_valid = (val_examples
                           .map(tf_encode,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
                           .filter(lambda pt, en: tf.logical_and(tf.size(pt) <= max_len,
                                                                 tf.size(en) <= max_len))
                           .padded_batch(batch_size, padded_shapes=([None], [None])))
