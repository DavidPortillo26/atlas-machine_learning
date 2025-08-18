#!/usr/bin/env python3
"""
Question Answering function using BERT
"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np

# Load model and tokenizer once to avoid reloading every function call
bert_model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question.

    Parameters:
        question (str): The question to answer
        reference (str): The reference document

    Returns:
        str or None: The answer snippet, or None if not found
    """

    # Tokenize question and reference together
    inputs = tokenizer.encode_plus(question, reference, return_tensors='tf')

    input_ids = inputs["input_ids"]
    input_mask = inputs["attention_mask"]

    # Run the model
    outputs = bert_model([input_ids, input_mask])
    start_scores, end_scores = outputs

    # Convert tensors to numpy
    start_scores = start_scores.numpy()[0]
    end_scores = end_scores.numpy()[0]

    # Get the most probable start and end token positions
    start_index = np.argmax(start_scores)
    end_index = np.argmax(end_scores)

    # Make sure the end index is after the start index
    if end_index < start_index:
        return None

    # Convert token ids to tokens
    tokens = input_ids[0][start_index : end_index + 1]
    answer_tokens = tokenizer.convert_ids_to_tokens(tokens.numpy())

    # Combine tokens into a string, cleaning up '##' subwords
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if answer.strip() == "":
        return None
    return answer
