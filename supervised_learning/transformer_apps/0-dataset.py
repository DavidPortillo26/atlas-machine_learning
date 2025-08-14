#!/usr/bin/env python3
import tensorflow_datasets as tfds
import transformers

class Dataset:
    """
    Loads and prepares the TED Talks Portuguese-English translation dataset
    with SubwordTextEncoder tokenizers built from the dataset itself.
    """

    def __init__(self):
        # Load the train and validation splits deterministically
        self.data_train, self.data_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split=["train", "validation"],
            as_supervised=True,
            shuffle_files=False
        )

        # Build tokenizers from the training dataset
        self.tokenizer_pt, self.tokenizer_en = self.tokenizer_dataset(self.data_train)

    def tokenizer_dataset(self, data):
        """
        Creates SubwordTextEncoder tokenizers for Portuguese and English.
        """

        # Limit to first 15000 examples for deterministic output
        def pt_gen():
            for i, (pt, en) in enumerate(data):
                if i >= 15000:
                    break
                yield pt.numpy().decode('utf-8')

        def en_gen():
            for i, (pt, en) in enumerate(data):
                if i >= 15000:
                    break
                yield en.numpy().decode('utf-8')

        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            pt_gen(), target_vocab_size=2**13
        )
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            en_gen(), target_vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
