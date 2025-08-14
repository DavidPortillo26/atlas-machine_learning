#!/usr/bin/env python3
import tensorflow_datasets as tfds
import transformers

class Dataset:
    def __init__(self):
        # Load the train and validation splits deterministically
        self.data_train, self.data_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split=["train", "validation"],
            as_supervised=True,
            shuffle_files=False
        )

        # Build tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenizer_dataset(self.data_train)

    def tokenizer_dataset(self, data):
        # Limit to first 10000 examples
        corpus = list(data.take(1000).as_numpy_iterator())

        pt_corpus = [pt.decode("utf-8") for pt, _ in corpus]
        en_corpus = [en.decode("utf-8") for _, en in corpus]

        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            pt_corpus, target_vocab_size=2**13
        )
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            en_corpus, target_vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
