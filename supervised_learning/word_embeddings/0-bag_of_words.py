#!/usr/bin/env python3
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag-of-words embedding matrix from a list of sentences.

    Args:
        sentences (list): List of sentences (strings) to analyze.
        vocab (list, optional): Vocabulary words to use. If None,
            all unique words from the sentences are used.

    Returns:
        tuple: (embeddings, features)
            embeddings (numpy.ndarray): Array of shape (s, f) with word frequencies.
            features (numpy.ndarray): Array of shape (f,) with the feature words.
    """

    def tokenize(text):
        """
        Tokenize a sentence into lowercase words.

        Uses regex to find all word sequences (alphanumeric and underscore),
        converts them to lowercase to normalize the tokens.

        Args:
            text (str): Sentence to tokenize.

        Returns:
            list: List of lowercase word tokens.
        """
        return re.findall(r'\b\w+\b', text.lower())

    # Tokenize each sentence in the input list
    tokenized_sentences = [tokenize(sentence) for sentence in sentences]

    # If vocab is not provided, build it from all unique words found in the tokenized sentences
    if vocab is None:
        all_words = set(
            word
            for sentence in tokenized_sentences
            for word in sentence
        )
        # Remove the possessive 's' which is common but usually uninformative
        all_words.discard('s')
        # Sort the vocabulary words to maintain consistent order and convert to numpy array
        features = np.array(sorted(all_words), dtype=object)
    else:
        # Use provided vocab; convert all words to lowercase for consistency
        features = np.array(
            [word.lower() for word in vocab],
            dtype=object
        )

    # Create a mapping from each vocabulary word to its index for quick lookup
    word_index = {word: idx for idx, word in enumerate(features)}

    # Initialize an embeddings matrix with zeros:
    # Rows = number of sentences, Columns = number of vocabulary words
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    # Count occurrences of each vocabulary word in each sentence
    for i, tokens in enumerate(tokenized_sentences):
        for word in tokens:
            if word in word_index:
                embeddings[i, word_index[word]] += 1

    # Return the embeddings matrix and the feature (vocabulary) list
    return embeddings, features
