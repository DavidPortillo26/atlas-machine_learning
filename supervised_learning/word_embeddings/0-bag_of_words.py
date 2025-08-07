#!/usr/bin/env python3
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag-of-words embedding matrix from a list of sentences.

    Args:
        sentences (list): A list of sentences (strings) to analyze.
        vocab (list, optional): A list of vocabulary words to use. If None,
                                all unique words from the sentences are used.

    Returns:
        tuple:
            - embeddings (numpy.ndarray): Array of shape (s, f) containing word frequencies.
            - features (numpy.ndarray): Array of shape (f,) containing the feature words.
    """
    def tokenize(text):
        """Tokenize a sentence into lowercase words."""
        return re.findall(r'\b\w+\b', text.lower())

    tokenized_sentences = [tokenize(sentence) for sentence in sentences]

    if vocab is None:
        # Generate vocabulary from all tokens
        all_words = set(
            word for sentence in tokenized_sentences for word in sentence
        )
        all_words.discard('s')  # Remove 's' (common artifact)
        features = np.array(sorted(all_words), dtype=object)
    else:
        # Use provided vocabulary in original order, lowercase
        features = np.array([word.lower() for word in vocab], dtype=object)

    word_index = {word: idx for idx, word in enumerate(features)}
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, tokens in enumerate(tokenized_sentences):
        for word in tokens:
            if word in word_index:
                embeddings[i, word_index[word]] += 1

    return embeddings, features
