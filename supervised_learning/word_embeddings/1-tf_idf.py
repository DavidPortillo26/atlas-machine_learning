#!/usr/bin/env python3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding for given sentences.

    Args:
        sentences (list of str): List of sentences.
        vocab (list of str): Vocabulary to use.If None, inferred from sentences.

    Returns:
        embeddings (np.ndarray): shape (s, f) TF-IDF matrix.
        features (list or np.ndarray): list of features used.
    """
    # Create vectorizer with optional fixed vocabulary
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    # Fit and transform the sentences
    X = vectorizer.fit_transform(sentences)

    # Get the feature names in order
    features = np.array(vectorizer.get_feature_names_out())

    # Convert to dense NumPy array
    embeddings = X.toarray()

    return embeddings, features
