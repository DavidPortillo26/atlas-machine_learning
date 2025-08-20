<img width="348" height="145" alt="image" src="https://github.com/user-attachments/assets/f1de91aa-9cc1-49cb-b3c2-18c1c13e5f9c" />


# Transformer Apps – Sequence-to-Sequence Learning with Transformers

![Transformer Illustration](https://raw.githubusercontent.com/DavidPortillo26/atlas-machine_learning/main/supervised_learning/transformer_apps/transformer_diagram.png)

## About Me

Hi! I’m **David Portillo**, a software engineer and machine learning enthusiast.
I love experimenting with deep learning models, natural language processing, and building practical AI applications.
You can connect with me on [LinkedIn](https://www.linkedin.com/in/david-portillo26/) or explore my [portfolio projects](https://github.com/DavidPortillo26/davidportillo.github.io).

---

## Project Overview

**Transformer Apps** is a collection of Python modules implementing **sequence-to-sequence (seq2seq) models** using Transformer architectures.
The project focuses on **machine translation**, specifically Portuguese-to-English translation with the TED HRLR dataset.

This project includes:

* Dataset loading and preprocessing
* Synthetic datasets for testing
* Tokenization and subword encoders
* Transformer padding and look-ahead masks

The project demonstrates **end-to-end workflow** from raw dataset to tokenized sequences ready for Transformer training.

---

## Features

### Dataset Modules

1. **0-dataset.py** – Loads the TED HRLR Portuguese-English dataset and builds subword tokenizers.
2. **1-dataset.py** – Provides a **small synthetic dataset** for testing tokenization and Transformer pipelines.
3. **2-dataset.py** – Converts synthetic sequences into TensorFlow tensors suitable for Transformer training.
4. **3-dataset.py** – Loads, encodes, and batches the TED HRLR dataset with proper padding and truncation for seq2seq models.

### Masking Module

5. **4-create\_masks.py** – Creates **padding masks** and **look-ahead masks** for encoder and decoder attention layers.

---

## Project Structure

```
transformer_apps/
├── 0-dataset.py        # TED HRLR dataset loader & tokenizer
├── 1-dataset.py        # Synthetic dataset generator
├── 2-dataset.py        # TensorFlow encoded sequences
├── 3-dataset.py        # Batched dataset for training
├── 4-create_masks.py   # Padding & look-ahead masks
```

---

## Usage Example

Here’s how to load the TED HRLR dataset, encode sequences, and create masks:

```python
from transformer_apps.3_dataset import Dataset
from transformer_apps.4_create_masks import create_masks
import tensorflow as tf

# Initialize dataset
batch_size = 64
max_len = 36
dataset = Dataset(batch_size=batch_size, max_len=max_len)

# Get one batch
for pt_batch, en_batch in dataset.data_train.take(1):
    print("Portuguese batch shape:", pt_batch.shape)
    print("English batch shape:", en_batch.shape)

# Create masks for a batch
encoder_mask, combined_mask, decoder_mask = create_masks(pt_batch, en_batch)
print("Encoder mask shape:", encoder_mask.shape)
print("Combined mask shape:", combined_mask.shape)
print("Decoder mask shape:", decoder_mask.shape)
```

---

## Development Story

During development, the main goals were:

1. Load the TED HRLR dataset efficiently.
2. Build **subword tokenizers** to handle rare words.
3. Create **synthetic datasets** to quickly test Transformer pipelines.
4. Implement **masking functions** to prevent the model from seeing padding or future tokens.

**Challenges encountered:**

* Efficient tokenization with TensorFlow Datasets.
* Designing the look-ahead mask to correctly combine padding and future-token masking.
* Ensuring synthetic datasets match expected professor examples for consistency.

---

## Features Implemented

* Dataset loading, encoding, batching ✅
* Synthetic dataset generation for testing ✅
* Tokenizer building with subword vocabulary ✅
* Padding and look-ahead masks for Transformers ✅

## Features To Implement

* Full Transformer training and inference loop
* BLEU score evaluation for translation quality
* Integration with GPU/TPU acceleration
* Interactive translation interface

---

## References

* TensorFlow Datasets – [TED HRLR](https://www.tensorflow.org/datasets/community_catalog/huggingface/ted_hrlr_translate)
* Vaswani et al., “Attention is All You Need”, 2017

---

If you want, I can **also add a visual diagram and step-by-step flow** of the dataset-to-mask pipeline to make this README even more engaging and clear for non-coders.

Do you want me to do that next?
