#!/usr/bin/env python3
"""
Question Answering function using BERT with token-based chunking
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
CHUNK_SIZE = 400  # token-based chunk size for reference text
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

    # Tokenize the entire reference first
    ref_tokens = tokenizer.tokenize(reference)
    total_tokens = len(ref_tokens)
    best_answer = None
    best_score = -np.inf

    # Process reference in overlapping token chunks
    for start in range(0, total_tokens, STEP):
        chunk_tokens = ref_tokens[start:start + CHUNK_SIZE]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)

        # Tokenize question + chunk
        inputs = tokenizer.encode_plus(
            question,
            chunk_text,
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

        # Ignore [CLS] predictions
        start_index = np.argmax(start_scores)
        end_index = np.argmax(end_scores)
        if start_index == 0 and end_index == 0:
            continue
        if end_index < start_index:
            continue

        # Calculate a confidence score (sum of start + end logits)
        score = start_scores[start_index] + end_scores[end_index]
        if score > best_score:
            best_score = score
            tokens = input_ids[0][start_index:end_index + 1]
            answer_tokens = tokenizer.convert_ids_to_tokens(tokens.numpy())
            best_answer = tokenizer.convert_tokens_to_string(answer_tokens).strip()

    return best_answer
