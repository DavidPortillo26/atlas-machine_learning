#!/usr/bin/env python3
"""Sort entries by their high price."""


def high(df):
    """
    Return the DataFrame sorted in descending order of the `High` column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing a `High` price column.

    Returns
    -------
    pandas.DataFrame
        Sorted DataFrame with highest prices first.
    """
    if 'High' not in df.columns:
        raise KeyError("DataFrame must have a 'High' column")

    return df.sort_values('High', ascending=False)
