#!/usr/bin/env python3
"""
Question Answering function using BERT with chunking for long references
"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np

# Load model and tokenizer once
bert_model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)

MAX_LEN = 512  # BERT max input length
CHUNK_SIZE = 400  # chunk size for reference text
STEP = 300        # overlap to avoid missing answers


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question.

    Parameters:
        question (str): The question to answer
        reference (str): The reference document

    Returns:
        str or None: The answer snippet, or None if not found
    """

    reference_len = len(reference)
    best_answer = None

    # Process the reference in overlapping chunks
    for start in range(0, reference_len, STEP):
        chunk = reference[start:start + CHUNK_SIZE]

        # Tokenize question and chunk
        inputs = tokenizer.encode_plus(
            question,
            chunk,
            return_tensors="tf",
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_token_type_ids=True,
        )

        input_ids = inputs["input_ids"]
        input_mask = inputs["attention_mask"]
        segment_ids = inputs["token_type_ids"]

        # Run the model
        outputs = bert_model([input_ids, input_mask, segment_ids])
        start_scores = outputs[0][0].numpy()
        end_scores = outputs[1][0].numpy()

        # Get the most probable start and end token positions
        start_index = np.argmax(start_scores)
        end_index = np.argmax(end_scores)

        # Ignore [CLS] predictions
        if start_index == 0 and end_index == 0:
            continue
        if end_index < start_index:
            continue

        # Convert token ids to tokens
        tokens = input_ids[0][start_index:end_index + 1]
        answer_tokens = tokenizer.convert_ids_to_tokens(tokens.numpy())

        # Combine tokens into a string, cleaning up '##' subwords
        answer = tokenizer.convert_tokens_to_string(answer_tokens).strip()

        if answer:
            best_answer = answer
            break  # return the first valid answer found

    return best_answer
