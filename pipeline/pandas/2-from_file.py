#!/usr/bin/env python3
"""Wrapper around pandas.read_csv for loading delimited text files."""

import pandas as pd


def from_file(filename, delimiter):
    """
    Read tabular data from a text file and return it as a pandas DataFrame.

    Parameters
    ----------
    filename : str
        The full name or path of the file that holds your data. Think of this
        as the location of your spreadsheet saved as plain text (for example,
        ``"data/customers.csv"``).
    delimiter : str
        The single character that separates one column from the next inside the
        file. Common examples are a comma (`,`) for CSV files or a tab
        (`\\t`) for TSV files.

    Returns
    -------
    pandas.DataFrame
        A DataFrame is pandas' version of a spreadsheet or table. It lets you
        look at the data, filter it, and perform calculations without editing
        the original file.

    Step-by-step:
        1. pandas opens the file located at `filename`.
        2. It looks at how the columns are separated using `delimiter`.
        3. It creates an in-memory table (the DataFrame) that you can work with.
    """
    return pd.read_csv(filename, sep=delimiter)
