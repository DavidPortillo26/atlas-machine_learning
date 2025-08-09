#!/usr/bin/env python3

import tensorflow_datasets as tfds
import transformers

data_train, data_valid = trfds.load(
    'ted_hrlr_translate/pt_to_en',
    split=['train', 'validation'],
    as_supervised=True
)

for pt, en in data_train.take(1):
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))


tokeniner_pt = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
tokeniner_en = BertTokenizer.from_pretrained('bert-base-uncased')

#Test them
Print(tokeniner_pt.tokenize("Olá, tudo bem?"))
Print(tokeniner_en.tokenize("Hello, how are you?"))

class Dataset:
    def __init__(self):
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )
        self.tokenizer_pt, self.tokenizer_en = self.tokenizer_dataset(self.data_train)
def tokenizer_dataset(self, data):
    tokenizer_pt = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
    tokeenizer_en = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer_pt, tokenizer_en
