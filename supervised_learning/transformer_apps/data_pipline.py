# data_pipeline.py
import tensorflow as tf
from transformers import AutoTokenizer

""" This file handles loading, parsing, and batching TFRecords"""

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def parse_tfrecord(example_proto):
    feature_description = {
        "input_ids": tf.io.FixedLenFeature([128], tf.int64),
        "attention_mask": tf.io.FixedLenFeature([128], tf.int64),
        "labels": tf.io.FixedLenFeature([128], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    return example["input_ids"], example["labels"]

def load_dataset(file_pattern, batch_size=32, shuffle_buffer_size=1000, cache=True):
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    
    if cache:
        dataset = dataset.cache()
    
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
