#!/usr/bin/env python3
"""
Module: 0-from_numpy
Provides `from_numpy(array)` to convert a
2D NumPy ndarray into a pandas DataFrame.
Columns are labeled 'A'..'Z' in order; at most 26 columns are supported.
"""
import pandas as pd


def from_numpy(array):
    """
    Create a pandas DataFrame from a NumPy array.

    Parameters
    ----------
    array : numpy.ndarray
        A 2D NumPy array where rows are observations and columns are features.

    Returns
    -------
    pd.DataFrame
        A DataFrame with column labels 'A', 'B', 'C', ... in order.

    Notes
    -----
    Assumes there are at most 26 columns (A–Z).
    """
    if getattr(array, "ndim", None) != 2:
        raise ValueError("from_numpy expects a 2D np.ndarray (rows, cols).")

    # Gets number of columns
    n_cols = array.shape[1]
    if n_cols > 26:
        raise ValueError("This function supports at most 26 columns (A–Z).")

    # Creates column labels
    columns = [chr(65 + i) for i in range(n_cols)]

    return pd.DataFrame(array, columns=columns)
