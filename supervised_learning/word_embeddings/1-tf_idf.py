#!/usr/bin/env python3
import numpy as np
import math

def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding.
    
    Args:
        sentences (list of str): Sentences to analyze.
        vocab (list of str): Vocabulary words to use. If None, extract from sentences.
    
    Returns:
        embeddings (np.ndarray): shape (s, f) containing the embeddings
        features (list of str): List of features (vocabulary words)
    """
    
    # Tokenize sentences into lists of words
    tokenized = [sentence.lower().split() for sentence in sentences]
    
    # Build vocabulary
    if vocab is None:
        vocab = sorted(set(word for sent in tokenized for word in sent))
    else:
        vocab = [word.lower() for word in vocab]
    
    features = vocab
    s = len(sentences)
    f = len(features)
    
    # Compute Term Frequency (TF)
    tf_matrix = np.zeros((s, f))
    for i, sent in enumerate(tokenized):
        for j, word in enumerate(features):
            tf_matrix[i, j] = sent.count(word) / len(sent) if len(sent) > 0 else 0
    
    # Compute Inverse Document Frequency (IDF)
    idf_vector = np.zeros(f)
    for j, word in enumerate(features):
        doc_count = sum(1 for sent in tokenized if word in sent)
        idf_vector[j] = math.log((1 + s) / (1 + doc_count)) + 1  # smooth to avoid div by zero
    
    # Compute TF-IDF
    embeddings = tf_matrix * idf_vector
    
    return embeddings, features
