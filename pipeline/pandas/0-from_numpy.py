#!/usr/bin/env python3

import pandas as pd

"""
    Create a pandas DataFrame from a NumPy array.

    Parameters
    ----------
    array : numpy.ndarray
        A 2D Numpy arrat where rows are observations and columns are features.

    Returns
    -------
    pd.DataFrame
        A Dataframe with column lables 'A', 'B', 'C', ... in order.

    Notes
    -----
    Assumes there are at most 26 columns (A - Z).
    """


def from_numpy(array):

    if getattr(array, "ndim", None) != 2:
        raise ValueError("from_numpy expects a 2D np.mdarray (rows, cols).")

    # Gets number of columns
    n_cols = array.shape[1]
    if n_cols > 26:
        raise ValueError(" This function supports at most 26 columns (A -Z). ")

    # Creates column labels
    columns = [chr(65 + i ) for i in range(n_cols)]
    return pd.DataFrame(array, columns=columns)
