#!/usr/bin/env python3
import gensim
import random
import numpy as np

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
    """
    # Ensure reproducibility
    random.seed(seed)
    np.random.seed(seed)

    sg = 0 if cbow else 1

    model = gensim.models.Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )

    # Build vocab first (deterministic)
    model.build_vocab(sentences)

    # Train model
    model.train(sentences, total_examples=len(sentences), epochs=epochs)

    return model
