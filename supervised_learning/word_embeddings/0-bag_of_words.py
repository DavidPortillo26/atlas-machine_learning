#!/usr/bin/env python3
import numpy as np
import re

def bag_of_words(sentences, vocab=None):
    def tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())

    tokenized_sentences = [tokenize(sentence) for sentence in sentences]

    if vocab is None:
        # Create vocab from all words in order of appearance (preserve order, remove duplicates)
        all_words = []
        seen = set()
        for sentence in tokenized_sentences:
            for word in sentence:
                if word not in seen:
                    seen.add(word)
                    all_words.append(word)
        features = np.array(all_words)
    else:
        features = np.array([word.lower() for word in vocab])

    word_index = {word: i for i, word in enumerate(features)}
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, tokens in enumerate(tokenized_sentences):
        for word in tokens:
            if word in word_index:
                embeddings[i, word_index[word]] += 1

    return embeddings, features
