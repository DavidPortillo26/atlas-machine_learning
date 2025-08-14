
#!/usr/bin/env python3
"""
    Transformer Applications
"""
import transformers
import tensorflow_datasets as tfds


class Dataset:
    """Dataset class loads and prepares a dataset for MT
    """
    def __init__(self) -> None:
        """Initializer
        """
        self.data_train, self.data_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split=['train', 'validation'],
            as_supervised=True
        )
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Creates subwords tokenizers for out dataset

        Arguments:
            data {tf.data.Dataset} -- Dataset of samples

        Returns:
            tuple -- Contains the tokenizer_pt and tokenizer_en
        """
        tok_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data),
            target_vocab_size=2**15
        )
        tok_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data),
            target_vocab_size=2**15
        )
        return tok_pt, tok_en