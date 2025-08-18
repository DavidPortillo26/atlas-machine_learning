#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_datasets as tfds

class Dataset:
    def __init__(self, batch_size, max_len):
        self.tokenizer_pt_vocab_size = 8192
        self.tokenizer_en_vocab_size = 8192
        self.batch_size = batch_size
        self.max_len = max_len

        # Load TED HRLR Portuguese-to-English dataset
        examples, metadata = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            as_supervised=True,
            with_info=True
        )
        train_examples, val_examples = examples['train'], examples['validation']

        # Build subword tokenizers
        self.tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.decode("utf-8") for pt, _ in tfds.as_numpy(train_examples)),
            target_vocab_size=self.tokenizer_pt_vocab_size
        )

        self.tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.decode("utf-8") for _, en in tfds.as_numpy(train_examples)),
            target_vocab_size=self.tokenizer_en_vocab_size
        )


        # Define start/end tokens
        self.start_token_pt = self.tokenizer_pt.vocab_size
        self.end_token_pt = self.tokenizer_pt.vocab_size + 1
        self.start_token_en = self.tokenizer_en.vocab_size
        self.end_token_en = self.tokenizer_en.vocab_size + 1

        # Encode function (vectorized)
        def encode(pt, en):
            pt = self.tokenizer_pt.encode(pt.numpy().decode("utf-8"))
            en = self.tokenizer_en.encode(en.numpy().decode("utf-8"))
            pt_tokens = [self.start_token_pt] + pt + [self.end_token_pt]
            en_tokens = [self.start_token_en] + en + [self.end_token_en]
            return tf.constant(pt_tokens, dtype=tf.int64), tf.constant(en_tokens, dtype=tf.int64)

        # Wrap for TensorFlow dataset
        def tf_encode(pt, en):
            pt_tokens, en_tokens = tf.py_function(
                encode, [pt, en], [tf.int64, tf.int64]
            )
            pt_tokens.set_shape([None])
            en_tokens.set_shape([None])
            return pt_tokens, en_tokens

        # Training dataset
        self.data_train = (
            train_examples
            .map(tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
            .filter(lambda pt, en: tf.logical_and(tf.size(pt) <= max_len, tf.size(en) <= max_len))
            .cache()
            .shuffle(20000)
            .padded_batch(
                batch_size,
                padded_shapes=([None], [None]),
                padding_values=(self.end_token_pt, self.end_token_en)
            )
            .prefetch(tf.data.AUTOTUNE)
        )

        # Validation dataset
        self.data_valid = (
            val_examples
            .map(tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
            .filter(lambda pt, en: tf.logical_and(tf.size(pt) <= max_len, tf.size(en) <= max_len))
            .padded_batch(
                batch_size,
                padded_shapes=([None], [None]),
                padding_values=(self.end_token_pt, self.end_token_en)
            )
        )
