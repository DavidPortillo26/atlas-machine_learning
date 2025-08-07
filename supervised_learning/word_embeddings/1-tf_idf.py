#!/usr/bin/env python3
"""tf-idf"""

from sklearn.feature_extraction.text import TfidfVectorizer

def tf_idf(sentences, vocab=None):

    """
    tf-idf function
    Args:
        sentences: list of sentences to be processed
        vocab: list of words to be used as vocabulary (default=None)
    Returns:
        embeddings: numpy.ndarray of shape (s, v) containing the embeddings
            s: number of sentences in sentences
            v: size of the vocabulary
    """
    if vocab is None:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)
        vocab = vectorizer.get_feature_names()
    else:
        vectorizer = TfidfVectorizer(vocabulary=vocab)
        X = vectorizer.fit_transform(sentences)
    embeddings = X.toarray()

    return embeddings, vocab
