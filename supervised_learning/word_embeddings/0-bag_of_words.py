#!/usr/bin/env python3
"""
    NLP - Word Embeddings
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix with preprocessing.
    
    Parameters:
    - sentences: list of strings
    - vocab: list of vocabulary words to use (optional)
    
    Returns:
    - embeddings: ndarray of shape (s, f)
    - features: list of feature names used
    """
    vectorizer = CountVectorizer(
        vocabulary=vocab,
        lowercase=True,
        token_pattern=r'\b\w\w+\b'  # excludes single-character tokens like 's'
    )
    
    embeddings = vectorizer.fit_transform(sentences)
    features = vectorizer.get_feature_names_out().tolist()
    
    return embeddings.toarray(), features
