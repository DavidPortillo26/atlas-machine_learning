#!/usr/bin/env python3
"""
Module for loading and tokenizing the TED HRLR Portuguese to English dataset.
"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Class to load and tokenize the TED HRLR translation dataset.

    Attributes
    ----------
    data_train : tf.data.Dataset
        Training dataset, loaded deterministically.
    data_valid : tf.data.Dataset
        Validation dataset, loaded deterministically.
    tokenizer_pt : tfds.deprecated.text.SubwordTextEncoder
        Subword tokenizer for Portuguese.
    tokenizer_en : tfds.deprecated.text.SubwordTextEncoder
        Subword tokenizer for English.
    """

    def __init__(self):
        """
        Initialize the Dataset class.

        Loads the training and validation splits of the TED HRLR
        Portuguese-to-English dataset, then builds subword tokenizers
        from the training data.
        """
        # Load the train and validation splits deterministically
        self.data_train, self.data_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split=["train", "validation"],
            as_supervised=True,
            shuffle_files=False
        )

        # Build tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenizer_dataset(
            self.data_train
        )

    def tokenizer_dataset(self, data):
        """
        Build subword tokenizers for Portuguese and English from data.

        Limits the corpus to the first 10,000 examples for faster training.

        Parameters
        ----------
        data : tf.data.Dataset
            Input dataset, typically the training dataset, as (pt, en) pairs.

        Returns
        -------
        tuple of tfds.deprecated.text.SubwordTextEncoder
            - tokenizer_pt : Subword tokenizer for Portuguese.
            - tokenizer_en : Subword tokenizer for English.
        """
        # Limit to first 10,000 examples
        corpus = list(data.take(10000).as_numpy_iterator())

        pt_corpus = [pt.decode("utf-8") for pt, _ in corpus]
        en_corpus = [en.decode("utf-8") for _, en in corpus]

        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            pt_corpus,
            target_vocab_size=2**13
        )
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            en_corpus,
            target_vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
