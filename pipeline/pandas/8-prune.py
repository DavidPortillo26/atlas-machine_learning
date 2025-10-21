#!/usr/bin/env python3
"""Drop rows lacking close prices."""


def prune(df):
    """
    Remove rows with missing values in the `Close` column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input market data that may contain missing close prices.

    Returns
    -------
    pandas.DataFrame
        DataFrame with all rows containing NaN in `Close` removed.
    """
    if 'Close' not in df.columns:
        raise KeyError("DataFrame must have a 'Close' column")

    return df.dropna(subset=['Close'])
