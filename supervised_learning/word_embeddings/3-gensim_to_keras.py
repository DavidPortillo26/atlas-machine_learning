#!/usr/bin/env python3
"""gensim to keras"""

def gensim_to_keras(model):
    """
    Convert a gensim Word2Vec model to a Keras Embedding layer.
    
    Args:
        model: Gensim Word2Vec model.
    
    Returns:
        embedding_layer: Keras Embedding layer initialized with the gensim model weights.
    """
    return modelwv.get_keras_embedding(train_embeddings=false)
