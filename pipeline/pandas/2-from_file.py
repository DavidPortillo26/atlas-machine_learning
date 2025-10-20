#!/usr/bin/env python3
"""
Module: 2-from_file

Provides a function `from_file(filename, delimiter)` that loads data from a file
into a pandas DataFrame.


Notes
-----
- This function uses pandas.read_csv() for loading files.
- It does not modify the input file and assumes it is well-formatted.
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Parameters
    ----------
    filename : str
        The path to the file to be read.
    delimiter : str
        The column separator used in the file (e.g., ',' for CSV, '\t' for TSV).

    Returns
    -------
    pandas.DataFrame
        The DataFrame containing the data loaded from the file.
    """
    # Uses pandas to read_csv with the provided delimiter
    df = pd.read_csv(filename, delimiter=delimiter)

    # Return the loaded DataFrame
    return df
