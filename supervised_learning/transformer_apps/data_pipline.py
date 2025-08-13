import tensorflow as tf

def create_dataset(file_paths, batch_size=32, buffer_size=10000):
    """
    Creates a TensorFlow dataset pipeline with proper cache, shuffle, batch, and repeat order.
    """

    # Load the dataset from TFRecord files (or any other source)
    dataset = tf.data.TFRecordDataset(file_paths)

    # Optional: parse TFRecords here
    def parse_fn(example):
        # Replace this with your actual parsing logic
        return example

    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=buffer_size)

    # Cache the dataset in memory for faster training
    dataset = dataset.cache()

    # Batch the dataset
    dataset = dataset.batch(batch_size)

    # Repeat the dataset indefinitely (for training loops)
    dataset = dataset.repeat()

    # Prefetch to improve pipeline performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
