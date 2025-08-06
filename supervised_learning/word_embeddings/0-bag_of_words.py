#!/usr/bin/enc python3
"""bag of words"""

from sklearn.feature_extraction.text import CountVectorizer

def bag_of_words(sentences, vocab=None):
    """
    bag of words function
    Args:
        sentences: list of sentences to be processed
        vocab: list of words to be used as vocabulary (default=None)
    Returns:
        embeddings: numpy.ndarray of shape (s, v) containing the embeddings
            s: number of sentences in sentences
            v: size of the vocabulary
    """
    if vocab is None:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(sentences)
        vocab = vectorizer.get_feature_names()
    else:
        vectorizer = CountVectorizer(vocabulary=vocab)
        X = vectorizer.fit_transform(sentences)
    embeddings = X.toarray()

    return embeddings, vocab
