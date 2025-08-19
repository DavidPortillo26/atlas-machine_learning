#!/usr/bin/env python3
"""
Question Answering function using BERT with token-based chunking and debug logging
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

MAX_LEN = 512   # BERT max input length
CHUNK_SIZE = 400  # token-based chunk size
STEP = 350        # overlap to avoid missing answers


def question_answer(question, reference):
    # Tokenize reference into chunks
    ref_tokens = tokenizer.tokenize(reference)
    best_answer = None
    best_score = -np.inf

    for i in range(0, len(ref_tokens), STEP):
        chunk_tokens = ref_tokens[i:i + CHUNK_SIZE]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)

        # Encode question + chunk
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

        # Run model
        outputs = bert_model([input_ids, input_mask, segment_ids])
        start_scores = outputs[0][0].numpy()
        end_scores = outputs[1][0].numpy()

        start_index = np.argmax(start_scores)
        end_index = np.argmax(end_scores)

        # Skip invalid spans
        if end_index < start_index or start_index == 0:
            continue

        score = start_scores[start_index] + end_scores[end_index]

        if score > best_score:
            tokens = input_ids[0][start_index:end_index + 1]
            answer_tokens = tokenizer.convert_ids_to_tokens(tokens.numpy())
            answer = tokenizer.convert_tokens_to_string(answer_tokens).strip()

            best_answer = answer
            best_score = score

    return best_answer if best_answer else None
