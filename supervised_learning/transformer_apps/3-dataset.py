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

        # Tokenizer
        self.tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in tfds.as_numpy(train_examples)),
            target_vocab_size=self.tokenizer_pt_vocab_size
        )
        self.tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in tfds.as_numpy(train_examples)),
            target_vocab_size=self.tokenizer_en_vocab_size
        )

        # Wrap encode in TF function
        def encode(pt, en):
            pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]
            en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(en.numpy()) + [self.tokenizer_en.vocab_size + 1]
            return tf.constant(pt_tokens, dtype=tf.int64), tf.constant(en_tokens, dtype=tf.int64)

        def tf_encode(pt, en):
            pt_tokens, en_tokens = tf.py_function(
                encode, [pt, en], [tf.int64, tf.int64]
            )
            pt_tokens.set_shape([None])
            en_tokens.set_shape([None])
            return pt_tokens, en_tokens

        # Prepare training data
        self.data_train = (train_examples
                           .map(tf_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                           .filter(lambda pt, en: tf.logical_and(tf.size(pt) <= max_len,
                                                                 tf.size(en) <= max_len))
                           .cache()
                           .shuffle(20000)
                           .padded_batch(batch_size, padded_shapes=([None], [None]))
                           .prefetch(tf.data.experimental.AUTOTUNE))

        # Prepare validation data
        self.data_valid = (val_examples
                           .map(tf_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                           .filter(lambda pt, en: tf.logical_and(tf.size(pt) <= max_len,
                                                                 tf.size(en) <= max_len))
                           .padded_batch(batch_size, padded_shapes=([None], [None])))
