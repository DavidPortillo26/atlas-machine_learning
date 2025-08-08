#!/usr/bin/env python3
def gensim_to_keras(model):
    """
    Converts a trained gensim Word2Vec model into a Keras Embedding layer.
    
    Args:
        model: Trained gensim Word2Vec model.
    
    Returns:
        A trainable keras.layers.Embedding layer initialized with gensim weights.
    """
    from tensorflow.keras.layers import Embedding
    import numpy as np
    
    # Get the number of words (vocabulary size) and embedding dimension
    vocab_size = len(model.wv)
    embedding_dim = model.vector_size
    
    # Extract weights matrix from gensim model
    weights = model.wv.vectors  # shape (vocab_size, embedding_dim)
    
    # Create a Keras Embedding layer with these weights
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[weights],
        trainable=True
    )
    
    return embedding_layer
