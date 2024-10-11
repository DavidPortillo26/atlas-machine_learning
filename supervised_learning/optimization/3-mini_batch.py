#!/usr/bin/env python3
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches for mini-batch gradient descent.

    This function divides a dataset into smaller groups, called mini-batches,
    which are used in mini-batch gradient descent to update model parameters in steps.
    Each mini-batch is a random subset of the dataset that ensures the model generalizes
    well by learning from varied portions of the data, while improving training efficiency
    compared to full-batch gradient descent.

    Args:
        X: numpy.ndarray of shape (m, nx)
           - Input data matrix where:
           - m is the number of data points (samples),
           - nx is the number of features for each data point (input dimensions).
        Y: numpy.ndarray of shape (m, ny)
           - Label matrix where:
           - m is the number of data points (same as in X),
           - ny is the number of output classes or targets
           in classification tasks (output dimensions).
        batch_size: int
           - The number of data points in each mini-batch.
           If the total number of data points
             is not divisible by batch_size, the last mini-batch
             will contain the remainder.

    Returns:
        List of mini-batches: Each mini-batch is a tuple (X_batch, Y_batch) where:
            - X_batch is a numpy.ndarray of shape (batch_size, nx),
            - Y_batch is a numpy.ndarray of shape (batch_size, ny).
        The function ensures that the final mini-batch may be smaller
        if the total number
        of data points is not divisible by batch_size.

    Function Details:
        1. **Shuffle the Data:**
           - Before creating mini-batches, the data is shuffled using
           the `shuffle_data` function. 
             Shuffling is important in stochastic gradient descent and its
             variants to avoid training on the data in a specific order. 
             This ensures the model doesn't overfit to patterns in sequential
             data points.

        2. **Calculate Total Data Points (`m`):**
           - The function calculates `m`, the total number of data points from
           the shape of `X`, to determine how many batches can be created.

        3. **Loop to Create Mini-Batches:**
           - The loop iterates over the entire dataset in steps of `batch_size`.
           In each iteration, a mini-batch of `batch_size` is extracted.
           - This is done by slicing `X` and `Y` from the current index `i` to `i +
           batch_size`. The result is a batch containing data points from that range.
           - These mini-batches are then stored in a list as tuples of `(X_batch, Y_batch)`.

        4. **Handling the Final Mini-Batch:**
           - If `m` is not divisible by `batch_size`, the final mini-batch will contain
           the remaining data points. For instance, if `m = 105` and `batch_size = 32`, 
             the final mini-batch will contain `105 % 32 = 9` data points.

    Example:
        Suppose you have the following:
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        Y = np.array([[1], [0], [1], [0], [1], [0]])
        batch_size = 2

        After calling create_mini_batches(X, Y, batch_size), you'll get:
        [
            (array([[3, 4], [1, 2]]), array([[0], [1]])),
            (array([[9, 10], [5, 6]]), array([[1], [1]])),
            (array([[11, 12], [7, 8]]), array([[0], [0]]))
        ]
        Each tuple contains a mini-batch of shuffled data points.
    """
    # Shuffle the data before creating batches to ensure randomization
    X, Y = shuffle_data(X, Y)

    # Number of data points in the dataset
    m = X.shape[0]
    
    # List to store mini-batches
    mini_batches = []

    # Loop through the data, creating mini-batches of size 'batch_size'
    for i in range(0, m, batch_size):
        # Select the next 'batch_size' data points for X and Y
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]

        # Append the batch as a tuple (X_batch, Y_batch) to the list of mini-batches
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
