#!/usr/bin/env python3

"""
Interactive question-answering from a corpus of reference documents.

This script lets a user ask questions. For each question:
1. It finds the most relevant document in a folder of text files.
2. It extracts the answer from that document using a previous QA function.
3. It responds to the user. The loop exits if the user types exit commands.
"""

import importlib.util
import sys
from typing import Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# Words that signal the program to stop asking questions
EXIT_COMMANDS = {"exit", "quit", "goodbye", "bye"}

# Dynamically load the previous single-document QA function
spec = importlib.util.spec_from_file_location("qa_module", "0-qa.py")
qa_module = importlib.util.module_from_spec(spec)
sys.modules["qa_module"] = qa_module
spec.loader.exec_module(qa_module)

# Get the function that finds an answer in a single reference text
question_answer_single = qa_module.question_answer

def semantic_search(corpus_path: str, sentence: str) -> str:
    """
    Find the document most similar in meaning to the question.

    Parameters:
        corpus_path (str): Folder containing text files as the corpus.
        sentence (str): The user's question.

    Returns:
        str: The text of the document most relevant to the question.
    """
    # Load a pre-trained model that understands semantic meaning
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Read all markdown files in the corpus directory
    corpus_texts = []
    for file in Path(corpus_path).glob("*.md"):
        text = file.read_text(encoding='utf-8')  # Read content of the file
        corpus_texts.append(text)                # Store it in a list

    # If no documents are found, return an empty string
    if not corpus_texts:
        return ""

    # Convert documents and the question into numerical embeddings
    corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True)
    query_embedding = model.encode(sentence, convert_to_tensor=True)

    # Compute similarity between the question and each document
    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)

    # Find the index of the most similar document
    best_idx = int(cosine_scores.argmax())

    # Return the content of the most relevant document
    return corpus_texts[best_idx]

def question_answer(corpus_path: str) -> None:
    """
    Continuously ask the user for questions and answer them using multiple 
    reference documents from a corpus.

    Parameters:
        corpus_path (str): Folder containing the reference documents.
    """
    while True:
        # Prompt user to type a question
        user_input = input("Q: ").strip()

        # Stop if user types an exit command
        if user_input.lower() in EXIT_COMMANDS:
            print("A: Goodbye!")
            break

        # Find the most relevant document from the corpus
        reference = semantic_search(corpus_path, user_input)

        # Use single-document QA function to extract the answer
        answer: Optional[str] = question_answer_single(user_input, reference)

        # Respond to the user
        if answer:
            print(f"A: {answer}")
        else:
            print("A: I'm sorry, I don't know the answer to that question.")
