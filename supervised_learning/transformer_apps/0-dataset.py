#!/usr/bin/env python3

import tensorflow_datasets as tfds
import transformers

class Dataset:
    def __init__(self):
        """Initializes the Dataset class by loading the TED Talks translation dataset
        and creating pretrained tokenizers for Portuguese and English."""
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )
        self.tokenizer_pt, self.tokenizer_en = self.tokenizer_dataset(self.data_train)
    
    def tokenizer_dataset(self, data):
        """
        Loads pretrained BERT tokenizers for Portuguese and English.

        Args:
            data: tf.data.Dataset of (pt, en) sentence pairs (not used for pretrained tokenizers)

        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained("neuralmind/bert-base-portuguese-cased")
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained("bert-base-uncased")
        return tokenizer_pt, tokenizer_en
