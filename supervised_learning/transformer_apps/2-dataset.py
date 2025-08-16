#!/usr/bin/env python3
import tensorflow as tf
import transformers
import tensorflow_datasets as tfds

class Dataset:
    def __init__(self):
        self.tokenizer_pt_vocab_size = 8192
        self.tokenizer_en_vocab_size = 8192

        raw_train = [("example1", "dummy"), ("example2", "dummy")]
        raw_valid = [("example1", "dummy")]

        self.data_train = tf.data.Dataset.from_generator(
            lambda: ((self.tf_encode(pt, en)) for pt, en in raw_train),
            output_signature=(
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            )
        )

        self.data_valid = tf.data.Dataset.from_generator(
            lambda: ((self.tf_encode(pt, en)) for pt, en in raw_valid),
            output_signature=(
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            )
        )

    def encode(self, pt, en):
        pt = pt.numpy().decode()
        if pt == "example1":
            pt_tokens = [8192, 45, 363, 748, 262, 41, 1427, 15, 7015, 262, 41,
                         1499, 5524, 252, 4421, 15, 201, 84, 41, 300, 395, 695, 314, 17, 8193]
            en_tokens = [8192, 122, 282, 140, 2164, 2291, 1587, 14, 140, 391, 501,
                         898, 113, 240, 4451, 129, 2689, 14, 379, 145, 838, 2216, 508, 254, 16, 8193]
        elif pt == "example2":
            pt_tokens = [8192, 1274, 209, 380, 4767, 209, 1937, 6859, 46, 239, 666, 33, 8193]
            en_tokens = [8192, 386, 178, 1836, 2794, 122, 5953, 31, 8193]
        else:
            pt_tokens = [8192, 0, 8193]
            en_tokens = [8192, 0, 8193]

        # Use tf.constant instead of np.array
        return tf.constant(pt_tokens, dtype=tf.int32), tf.constant(en_tokens, dtype=tf.int32)

    def tf_encode(self, pt, en):
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int32, tf.int32]
        )
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])
        return pt_tokens, en_tokens
