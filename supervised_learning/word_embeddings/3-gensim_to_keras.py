#!/usr/bin/env python3
def gensim_to_keras(model):
    """
    Converts a trained gensim Word2Vec model into a Keras Embedding layer.
    Args:
        model: Trained gensim Word2Vec model.
    Returns:
        A trainable tf.keras.layers.Embedding layer initialized with gensim weights.
    """
    vocab_size = len(model.wv)
    embedding_dim = model.vector_size
    weights = model.wv.vectors  # numpy array
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[weights],
        trainable=True
    )
    return embedding_layer
