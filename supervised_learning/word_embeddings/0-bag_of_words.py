#!/usr/bin/env python3
import numpy as np
import re

def bag_of_words(sentences, vocab=None):
    def tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())

    tokenized_sentences = [tokenize(sentence) for sentence in sentences]

    if vocab is None:
        all_words = set(word for sentence in tokenized_sentences for word in sentence)
        all_words.discard('s')  # remove 's' if generated from scratch
        features = np.array(sorted(all_words), dtype=object)
    else:
        # Keep original order and normalize to lowercase
        features = np.array([word.lower() for word in vocab], dtype=object)

    word_index = {word: i for i, word in enumerate(features)}
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, tokens in enumerate(tokenized_sentences):
        for word in tokens:
            if word in word_index:
                embeddings[i, word_index[word]] += 1

    return embeddings, features
