import numpy as np
import re

def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences (list): A list of sentences (strings).
        vocab (list or None): A list of vocabulary words to use. If None, generate from sentences.

    Returns:
        embeddings (numpy.ndarray): Shape (s, f), where s = # of sentences, f = # of features (words).
        features (list): The vocabulary used (in order).
    """
    # Helper function to tokenize sentences
    def tokenize(text):
        # Lowercase and remove non-alphanumeric characters (basic preprocessing)
        return re.findall(r'\b\w+\b', text.lower())

    # Tokenize all sentences
    tokenized_sentences = [tokenize(sentence) for sentence in sentences]

    # Create vocabulary if not provided
    if vocab is None:
        # Flatten all tokens and get unique sorted set
        vocab_set = sorted(set(word for sentence in tokenized_sentences for word in sentence))
    else:
        # Normalize provided vocab
        vocab_set = sorted(set(word.lower() for word in vocab))

    # Create word index mapping
    word_index = {word: idx for idx, word in enumerate(vocab_set)}

    # Initialize embedding matrix
    s = len(sentences)
    f = len(vocab_set)
    embeddings = np.zeros((s, f), dtype=int)

    # Fill the embedding matrix
    for i, tokens in enumerate(tokenized_sentences):
        for word in tokens:
            if word in word_index:
                embeddings[i, word_index[word]] += 1

    return embeddings, vocab_set
