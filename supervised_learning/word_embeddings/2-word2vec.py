#!/usr/bin/env python3
import gensim

def word2vec_model(
    sentences,
    vector_size=100,
    min_count=5,
    window=5,
    negative=5,
    cbow=True,
    epochs=5,
    seed=0,
    workers=1
):
    """
    Creates, builds, and trains a gensim Word2Vec model.

    Args:
        sentences (list): List of tokenized sentences to train on.
        vector_size (int): Dimensionality of the embedding vectors.
        min_count (int): Minimum count threshold for words.
        window (int): Max distance between current and predicted word.
        negative (int): Number of negative samples.
        cbow (bool): True for CBOW, False for Skip-gram.
        epochs (int): Number of training epochs.
        seed (int): Random seed.
        workers (int): Number of worker threads.

    Returns:
        gensim.models.Word2Vec: The trained Word2Vec model.
    """
    sg = 0 if cbow else 1  # CBOW if sg=0, Skip-gram if sg=1

    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )

    model.train(sentences, total_examples=len(sentences), epochs=epochs)
    return model
