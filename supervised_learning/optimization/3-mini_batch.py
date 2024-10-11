#!/usr/bin/env python3
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches for mini-batch gradient descent.

    Args:
        X: numpy.ndarray of shape (m, nx), input data
           m is the number of data points
           nx is the number of features in X
        Y: numpy.ndarray of shape (m, ny), labels
           m is the same number of data points as in X
           ny is the number of classes (or targets) for classification tasks
        batch_size: number of data points per batch

    Returns:
        List of mini-batches, where each mini-batch 
        is a tuple (X_batch, Y_batch)
    """
    # Shuffle the data
    X, Y = shuffle_data(X, Y)

    m = X.shape[0]  # Number of data points
    mini_batches = []

    # Create mini-batches
    for i in range(0, m, batch_size):
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
