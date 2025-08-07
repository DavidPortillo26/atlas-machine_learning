#!/usr/bin/env python3
"""
    NLP - Word Embeddings
"""
import numpy as np
from gensim.models import fasttext


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """ creates and trains a genism fastText model:

        sentences is a list of sentences to be trained on
        size is the dimensionality of the embedding layer
        min_count minimum number of occurrences of a word for use in training
        window is the maximum distance between the current word and predicted
        negative is the size of negative sampling
        cbow boolean to determine the training type
            True is for CBOW; False is for Skip-gram
        iterations is the number of iterations to train over
        seed is the seed for the random number generator
        workers is the number of worker threads to train the model
        Returns: the trained model
    """
    model = fasttext.FastText(sentences=sentences, size=size,
                              window=window, seed=seed, cbow_mean=int(cbow),
                              min_count=min_count, workers=workers,
                              iter=iterations)
    return model
