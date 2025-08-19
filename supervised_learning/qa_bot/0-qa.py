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
    inputs = tokenizer.encode_plus(
        question,
        reference,
        return_tensors="tf",
        max_length=512,
        truncation=True,
        padding="max_length",
        return_token_type_ids=True,
    )
    input_ids = inputs["input_ids"]
    input_mask = inputs["attention_mask"]
    segment_ids = inputs["token_type_ids"]

    outputs = bert_model([input_ids, input_mask, segment_ids])
    start_scores = outputs[0][0].numpy()
    end_scores = outputs[1][0].numpy()

    start_index = np.argmax(start_scores)
    end_index = np.argmax(end_scores)

    # Remove [CLS] token check
    if end_index < start_index:
        return None

    tokens = input_ids[0][start_index:end_index + 1]
    answer_tokens = tokenizer.convert_ids_to_tokens(tokens.numpy())
    answer = tokenizer.convert_tokens_to_string(answer_tokens).strip()

    return answer if answer else None
