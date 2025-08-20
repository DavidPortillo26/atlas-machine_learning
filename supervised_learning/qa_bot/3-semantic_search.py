#!/usr/bin/env python3

"""
Performs semantic search on corpus of documents

Given a sentence, finds the document in corpus that is most similar to the sentence based on semantic meaning
"""

import os
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

def semantic_search(corpus_path: str, sentence: str) -> str:
    """
    Finds the most semantically similar document in the corpus to the given sentence.

    Parameters:
    corpus_path (str): Path to the directory containing text files as corpus.
    sentence (str): The input sentence to compare against the corpus.

    Returns:
    str: The most similar document from the corpus.
    """
    # Load pre-trained SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Read all text files from the corpus directory
    corpus_texts = []
    file_paths = []
    for file in Path(corpus_path).glob("*.md"):
        text = file.read_text(encoding='utf-8')
        corpus_texts.append(text)
        file_paths.append(file)

    if not corpus_texts:
        return ""

    #encode corpus and query sentence
    corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True)
    query_embedding = model.encode(sentence, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)

    # Find the index of the highest scoring document
    best_idx = int(cosine_scores.argmax())

    # Return the most similar document
    return corpus_texts[best_idx]
