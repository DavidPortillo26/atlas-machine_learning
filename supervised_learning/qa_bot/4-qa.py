#!/usr/bin/env python3

"""
Interactive question-answering from a corpus of reference documents.

- Uses semantic search to select the most relevant document for each question.
- Uses the previous `question_answer` function to extract the answer from
  the selected document.
- Exits the loop when the user types exit, quit, goodbye, or bye.
"""

import importlib.util
import sys
from typing import Optional
from pathlib import Path

# Dynamically import the 0-qa module
spec = importlib.util.spec_from_file_location("qa_module", "0-qa.py")
qa_module = importlib.util.module_from_spec(spec)
sys.modules["qa_module"] = qa_module
spec.loader.exec_module(qa_module)

#import the existing question_answer function
question_answer_single = qa_module.question_answer

from sentence_transformers import SentenceTransformer, util

EXIT_COMMANDS = {"exit", "quit", "goodbye", "bye"}

def semantic_search(corpus_path: str, sentence: str) -> str:
    """
    Finds the document most semantically simalr to the given sentence.
    from all .md file in the corpus_path.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    corpus_texts = []
    for file in Path(corpus_path).glob("*.md"):
        text = file.read_text(encoding='utf-8')
        corpus_texts.append(text)

    if not corpus_texts:
        return ""

    corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True)
    query_embedding = model.encode(sentence, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)
    best_idx = int(cosine_scores.argmax())
    return corpus_texts[best_idx]

def question_answer(corpus_path: str) -> None:
    """
    Conitnuously prompts the user for questions and answers using
    muliple reference texts from corpus.
    """
    while True:
        user_input = input("Q: ").strip()
        if user_input.lower() in EXIT_COMMANDS:
            print("A: Goodbye!")
            break

        # FInd the most relevant document form the corpus
        reference = semantic_search(corpus_path, user_input)

        # Use the previous single document QA function
        answer : Optional[str] = question_answer_single(user_input, reference)
        if answer:
            print(f"A: {answer}")
        else:
            print("A: I'm sorry, I don't know the answer to that question.")
