#!/usr/bin/env python3
import numpy as np
import tensorflow_datasets as tfds
import transformers

class Dataset:
    def __init__(self):
        # Load data
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )
        # Load tokenizers
        self.tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        self.tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            "bert-base-uncased"
        )

    def encode(self, pt, en):
        """
        converts the tensorflow tensors to utf-8 strings
        args:
            .numpy(): converts the tensor to a bytes object
            .decorde('utf-8'): converts the bytes object to a normal string
        """
        pt_str =pt.numpy().decode('utf-8')
        en_str =en.numpy().decode('utf-8')
        """
        Tokenizer the sentences using the pretrained BERT tokenizers
        args:
            .encode(): method of the tokenizer that converts a string to a list of token ids
            pt_ids: list of token ids for the Portuguese sentence
            en_ids: list of token ids for the English sentence
        """
        pt_ids = self.tokenizer_pt.encode(pt_str)
        en_ids = self.tokenizer_en.encode(en_str)

        """
        Add start and end tokens to the tokenized sentences
        args:
            .tokenizer_pt: length of the Portuguese tokenizer vocabulary
            .tokenizer_en: length of the English tokenizer vocabulary
            pt_start: index of the start token for Portuguese
            pt_end: index of the end token for Portuguese
            en_start: index of the start token for English
            en_end: index of the end token for English
        """
        pt_start = len(self.tokenizer_pt)
        pt_end = len(self.tokenizer_pt) + 1
        en_start = len(self.tokenizer_en)
        en_end = len(self.tokenizer_en) + 1

        #Added the start and end tokens to the tokenized lists
        pt_tokens = [pt_start] + pt_ids + [pt_end]
        en_tokens = [en_start] + en_ids + [en_end]

        #Returns the tokenized sentences as numpy arrays
        pt_tokens = np.array(pt_tokens)
        en_tokens = np.array(en_tokens)

        return pt_tokens, en_tokens
