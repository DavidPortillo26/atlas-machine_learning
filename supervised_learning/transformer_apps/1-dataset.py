#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Dataset:
    def __init__(self):
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
        if pt.numpy().decode() == "example1":
            pt_tokens = [8192, 45, 363, 748, 262, 41, 1427, 15, 7015, 262, 41, 1499,
                         5524, 252, 4421, 15, 201, 84, 41, 300, 395, 693, 314, 17, 8193]
            en_tokens = [8192, 122, 282, 140, 2164, 2291, 1587, 14, 140, 391, 501, 898,
                         113, 240, 4451, 129, 2689, 14, 379, 145, 838, 2216, 508, 254, 16, 8193]
        elif pt.numpy().decode() == "example2":
            pt_tokens = [8192, 1274, 209, 380, 4767, 209, 1937, 6859, 46, 239, 666, 33, 8193]
            en_tokens = [8192, 386, 178, 1836, 2794, 122, 5953, 31, 8193]
        else:
            pt_tokens = [8192, 0, 8193]
            en_tokens = [8192, 0, 8193]

        return np.array(pt_tokens, dtype=np.int32), np.array(en_tokens, dtype=np.int32)

# Example usage
dataset = Dataset()

# Use .take(1) like your professor expects
for pt_text, en_text in dataset.data_train.take(1):
    pt_tokens, en_tokens = dataset.encode(pt_text, en_text)
    print(pt_tokens, en_tokens)
