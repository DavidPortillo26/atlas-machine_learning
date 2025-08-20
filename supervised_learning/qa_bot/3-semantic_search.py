#!/usr/bin/env python3

"""
Perform semantic search on a corpus of documents.

Given a sentence, this code finds the document in a corpus that is most
similar in meaning to the sentence, using a pre-trained language model.
"""

import os
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

def semantic_search(corpus_path: str, sentence: str) -> str:
    """
    Find the document most similar in meaning to the input sentence.

    Parameters:
        corpus_path (str): Directory containing text documents as the corpus.
        sentence (str): The sentence to search for within the corpus.

    Returns:
        str: The text of the document that is most similar to the sentence.
    """
    # Load a pre-trained SentenceTransformer model that understands text meaning
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Read all markdown files in the corpus directory
    corpus_texts = []
    file_paths = []
    for file in Path(corpus_path).glob("*.md"):
        text = file.read_text(encoding='utf-8')  # Read the content of the file
        corpus_texts.append(text)                # Save the content
        file_paths.append(file)                  # Keep track of file paths

    # If no documents are found, return an empty string
    if not corpus_texts:
        return ""

    # Convert all documents and the query sentence into numerical embeddings
    corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True)
    query_embedding = model.encode(sentence, convert_to_tensor=True)

    # Compute similarity between the query sentence and each document
    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)

    # Find the index of the document with the highest similarity score
    best_idx = int(cosine_scores.argmax())

    # Return the text of the document most similar to the query
    return corpus_texts[best_idx]
