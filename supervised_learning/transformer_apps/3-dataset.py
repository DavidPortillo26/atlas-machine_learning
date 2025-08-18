#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset:
    def __init__(self, batch_size, max_len):
        """
        Initializes the Dataset pipeline
        batch_size: size of each training/validation batch
        max_len: maximum number of tokens per sentence
        """

        self.tokenizer_pt_vocab_size = 8192
        self.tokenizer_en_vocab_size = 8192

        # For this project we load a real dataset (TED Talks Translation: Portuguese ↔ English)
        # as that's what your professor is expecting to see
        examples, metadata = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            with_info=True,
            as_supervised=True
        )
        train_examples, val_examples = examples['train'], examples['validation']

        # Map sentences → tokens
        self.data_train = train_examples.map(self.tf_encode)
        self.data_valid = val_examples.map(self.tf_encode)

        # Training pipeline
        self.data_train = (
            self.data_train
            .filter(lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len,
                tf.size(en) <= max_len
            ))
            .cache()
            .shuffle(20000)
            .padded_batch(batch_size, padded_shapes=([None], [None]))
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        # Validation pipeline
        self.data_valid = (
            self.data_valid
            .filter(lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len,
                tf.size(en) <= max_len
            ))
            .padded_batch(batch_size, padded_shapes=([None], [None]))
        )

    def encode(self, pt, en):
        """
        Encode sentences into token IDs.
        In a real project you'd use a subword tokenizer,
        but here we rely on the built-in TED dataset's subwords tokenizer.
        """
        pt_tokens = self.tokenizer_pt.encode(pt.numpy())
        en_tokens = self.tokenizer_en.encode(en.numpy())
        return (
            [self.tokenizer_pt_vocab_size] + pt_tokens + [self.tokenizer_pt_vocab_size + 1],
            [self.tokenizer_en_vocab_size] + en_tokens + [self.tokenizer_en_vocab_size + 1]
        )

    def tf_encode(self, pt, en):
        """
        Wraps the Python encode() so it can run inside TensorFlow graph
        """
        pt_tokens, en_tokens = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])
        return pt_tokens, en_tokens


# Load subword tokenizers once (static variables on the class)
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, _ in examples['train']),
    target_vocab_size=2**13
)
tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for _, en in examples['train']),
    target_vocab_size=2**13
)
Dataset.tokenizer_pt = tokenizer_pt
Dataset.tokenizer_en = tokenizer_en
