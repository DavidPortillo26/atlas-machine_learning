#!/usr/bin/env python3
"""
Defines function that finds a snippet of text within a reference document
to answer a question
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow_hub")

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer

# Load once
tokenizer = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad'
)
model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question
    """
    quest_tokens = tokenizer.tokenize(question)
    refer_tokens = tokenizer.tokenize(reference)

    # truncate if too long for BERT
    if len(quest_tokens) + len(refer_tokens) + 3 > 512:
        refer_tokens = refer_tokens[:512 - len(quest_tokens) - 3]

    all_tokens = ['[CLS]'] + quest_tokens + ['[SEP]'] + refer_tokens + ['[SEP]']

    input_ids = tokenizer.convert_tokens_to_ids(all_tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * (len(quest_tokens) + 2) + [1] * (len(refer_tokens) + 1)

    # convert to tensors
    input_ids, input_mask, segment_ids = map(
        lambda t: tf.expand_dims(
            tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_ids, input_mask, segment_ids))

    outputs = model([input_ids, input_mask, segment_ids])

    start_logits = outputs[0][0].numpy()
    end_logits = outputs[1][0].numpy()

    # Skip the first token ([CLS]) to avoid returning it
    start_index = start_logits[1:].argmax() + 1
    end_index = end_logits[1:].argmax() + 1

    if start_index >= len(all_tokens) or end_index >= len(all_tokens):
        return None
    if start_index > end_index:
        return None

    answer_tokens = all_tokens[start_index:end_index + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens).strip()

    # Return None if answer is empty
    if not answer:
        return None

    return answer
