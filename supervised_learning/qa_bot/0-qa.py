#!/usr/bin/env python3
"""
Question Answering function using BERT.

This module defines a function `question_answer` that can find the answer to a
question from a longer reference text. 

It uses a pre-trained BERT model fine-tuned on question-answering tasks 
(SQuAD dataset) via TensorFlow Hub. The function will return the text snippet
that answers the question, or None if no answer is found.

All TensorFlow Hub warnings are suppressed to keep the output clean.
"""

import warnings

# Ignore warnings from TensorFlow Hub about deprecated packages
warnings.filterwarnings(
    "ignore", category=UserWarning, module="tensorflow_hub"
)

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer

# Load the BERT tokenizer once. The tokenizer converts words into numbers 
# (tokens) that the BERT model can understand.
tokenizer = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad'
)

# Load the BERT QA model from TensorFlow Hub once. This model can take a 
# question and a reference text and predict which part of the text answers
# the question.
model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")


def question_answer(question, reference):
    """
    Find a snippet in the reference text that answers the given question.

    Parameters:
        question (str): The question you want to ask.
        reference (str): The text where the answer might be.

    Returns:
        str or None: Returns the answer as a string, or None if not found.
    """

    # Convert the question into tokens (numbers representing words)
    quest_tokens = tokenizer.tokenize(question)

    # Convert the reference text into tokens
    refer_tokens = tokenizer.tokenize(reference)

    # BERT has a maximum input length of 512 tokens. If the combined length
    # of the question and reference is too long, truncate the reference text
    if len(quest_tokens) + len(refer_tokens) + 3 > 512:
        refer_tokens = refer_tokens[:512 - len(quest_tokens) - 3]

    # Create the final token sequence for BERT:
    # [CLS] token at start, question tokens, [SEP] separator, reference tokens, [SEP]
    all_tokens = ['[CLS]'] + quest_tokens + ['[SEP]'] + refer_tokens + ['[SEP]']

    # Convert tokens to numeric IDs that BERT understands
    input_ids = tokenizer.convert_tokens_to_ids(all_tokens)

    # Create an attention mask: 1 for real tokens, 0 for padding (none here)
    input_mask = [1] * len(input_ids)

    # Create segment IDs: 0 for question, 1 for reference text
    segment_ids = [0] * (len(quest_tokens) + 2) + [1] * (len(refer_tokens) + 1)

    # Convert lists into tensors and add a batch dimension for the model
    input_ids, input_mask, segment_ids = map(
        lambda t: tf.expand_dims(tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_ids, input_mask, segment_ids)
    )

    # Run the model. It predicts two sets of numbers (logits):
    # start_logits indicates the start position of the answer
    # end_logits indicates the end position of the answer
    outputs = model([input_ids, input_mask, segment_ids])
    start_logits = outputs[0][0].numpy()
    end_logits = outputs[1][0].numpy()

    # Skip the first token ([CLS]) to avoid returning it as an answer
    start_index = start_logits[1:].argmax() + 1
    end_index = end_logits[1:].argmax() + 1

    # Check if the predicted indices are valid
    if start_index >= len(all_tokens) or end_index >= len(all_tokens):
        return None
    if start_index > end_index:
        return None

    # Extract the tokens corresponding to the predicted answer
    answer_tokens = all_tokens[start_index:end_index + 1]

    # Convert the tokens back into readable text
    answer = tokenizer.convert_tokens_to_string(answer_tokens).strip()

    # If the answer is empty, return None
    if not answer:
        return None

    return answer
