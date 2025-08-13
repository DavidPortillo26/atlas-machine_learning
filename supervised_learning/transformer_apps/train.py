#!/usr/bin/env python3
from transformer_apps.dataset import Dataset

"""This file handles model defention, training loops, and uses data_pipline.py to load the data."""

train_dataset = load_dataset("data/train/*.tfrecord")
val_dataset = load_dataset("data/val/*.tfrecord")