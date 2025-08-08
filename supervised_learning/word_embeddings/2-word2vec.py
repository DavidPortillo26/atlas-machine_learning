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
    sg = 0 if cbow else 1  # CBOW=0, Skipgram=1
    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )
    model.train(sentences, total_examples=len(sentences), epochs=epochs)
    return model
