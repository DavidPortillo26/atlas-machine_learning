#!/usr/bin/env python3
"""Reverse-chronological transpose of market data."""


def flip_switch(df):
    """
    Sort the DataFrame in descending order by index and transpose it.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data indexed by timestamp or another sortable label.

    Returns
    -------
    pandas.DataFrame
        A transposed DataFrame reflecting reverse chronological order.
    """
    if 'Timestamp' in df.columns:
        # Sorting by the explicit timestamp column is the safest interpretation
        # of "reverse chronological order".
        sorted_df = df.sort_values('Timestamp', ascending=False)
    else:
        sorted_df = df.sort_index(ascending=False)

    return sorted_df.transpose()
