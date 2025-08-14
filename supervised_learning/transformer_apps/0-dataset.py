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

    def tokenizer_dataset(self, data, max_examples=5000):
        """
        Creates SubwordTextEncoder tokenizers for Portuguese and English
        using only a limited number of examples to avoid timeout.
        """
        def limited_corpus(lang_index):
            count = 0
            for pt, en in data:
                text = pt.numpy().decode('utf-8') if lang_index == 0 else en.numpy().decode('utf-8')
                yield text
                count += 1
                if count >= max_examples:
                    break

        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            limited_corpus(0), target_vocab_size=2**13
        )
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            limited_corpus(1), target_vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
