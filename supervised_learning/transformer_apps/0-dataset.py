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
        # Materialize dataset so we can iterate multiple times
        corpus = list(data.as_numpy_iterator())

        # Decode tensors to strings
        pt_corpus = (pt.decode('utf-8') for pt, en in corpus[:10000])
        en_corpus = (en.decode('utf-8') for pt, en in corpus[:10000])

        # Build subword tokenizers
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            pt_corpus, target_vocab_size=2**13
        )
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            en_corpus, target_vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
