#!/usr/bin/env python3

"""word2vec"""

def word2vec_model(sentences, size=100, min_count=5,
                    window=5, negative=5, cbow=true,
                    iterations=5, seed=0, workers=1):
    """
    word2vec model function
    Args:
        sentences: list of sentences to be processed
        size: dimensionality of the word vectors (default=100)
        min_count: minimum word count to consider a word (default=5)
        window: maximum distance between the current and predicted word
            within a sentence (default=5)
        negative: number of negative samples to use (default=5)
        cbow: whether to use CBOW or skip-gram model (default=True)
        iterations: number of iterations over the corpus (default=5)
        seed: random seed for reproducibility (default=0)
        workers: number of worker threads to train the model (default=1)
    Returns:
        model: trained word2vec model
    """
    model = gensim.models.Word2Vec(sentences, min_count=count=min_count,
                                    iter=iterations, size=size,
                                    window=window, negative=negative,
                                    seed=seed, workers=workers, sg=cbow)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.iter)
    
    return model
