#!/usr/bin/env python3
"""
Module: 2-from_file

Provides `from_file(filename, delimiter)` to load delimited text files into a
pandas DataFrame, with automatic encoding fallback (utf-8 → latin-1 → cp1252).
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Load data from a file as a pandas DataFrame.

    Parameters
    ----------
    filename : str
        Path to the file to be read.
    delimiter : str
        Column separator used in the file (e.g., ',' for CSV, '\\t' for TSV).

    Returns
    -------
    pandas.DataFrame
        The DataFrame containing the data loaded from the file.

    Notes
    -----
    Tries UTF-8 first; on UnicodeDecodeError, retries with latin-1 and cp1252.
    """
    # Try UTF-8 first (standard). If that fails, fall back to common legacy encodings.
    encodings_to_try = ["utf-8", "latin-1", "cp1252"]
    last_err = None

    for enc in encodings_to_try:
        try:
            # low_memory=False: better type inference for wide/mixed CSVs
            return pd.read_csv(
                filename,
                delimiter=delimiter,
                encoding=enc,
                low_memory=False,
            )
        except UnicodeDecodeError as e:
            last_err = e
            continue

    # If all encodings failed, re-raise the last error with context.
    raise UnicodeDecodeError(
        last_err.encoding, last_err.object, last_err.start, last_err.end,
        f"{last_err.reason}. Tried encodings: {', '.join(encodings_to_try)}"
    )
