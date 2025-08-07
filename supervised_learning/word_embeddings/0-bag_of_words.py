#!/usr/bin/env python3
"""
    NLP - Word Embeddings
"""
import numpy as np
from sklearn.feature_extraction import text


def bag_of_words(sentences, vocab=None):
    """ Creates a bag of words embedding matrix
        sentences is a list of sentences to analyze
        vocab is a list of the vocabulary words to use for the analysis
            If None, all words within sentences should be used
        Returns: embeddings, features
            embeddings ndarray (s, f) containing the embeddings
                s is the number of sentences in sentences
                f is the number of features analyzed
            features list of the features used for embeddings
    """
    # vocab = ['are', 'awesome', 'beautiful', 'cake']
    Vectorizer = text.CountVectorizer(vocabulary=vocab)
    embeddings = Vectorizer.fit_transform(sentences)
    return embeddings.toarray(), Vectorizer.get_feature_names()
    """ doing from scratch not necessary
    features = set()
    embeddings = [[0]]
    for sentence in sentences:
        # new = set(s.replace("'s","").strip('@!#$%^&*()\'\" ').\
                lower() for s in sentence.split())
        # print(new)
        for n in new:
            features.add(n)
        #print(features)
    features = list(features)
    features.sort()
    return embeddings, features
    """
