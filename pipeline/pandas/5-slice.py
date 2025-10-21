#!/usr/bin/env python3
"""Select hourly snapshots of price and volume data."""


def slice(df):
    """
    Return every 60th row of the High, Low, Close, and volume columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing price and volume information.

    Returns
    -------
    pandas.DataFrame
        A DataFrame filtered to the requested columns with every 60th row.
    """
    required = ['High', 'Low', 'Close']
    for column in required:
        if column not in df.columns:
            raise KeyError(f"DataFrame must have a '{column}' column")

    volume_column = None
    if 'Volume_BTC' in df.columns:
        volume_column = 'Volume_BTC'
    elif 'Volume_(BTC)' in df.columns:
        volume_column = 'Volume_(BTC)'
    else:
        raise KeyError("DataFrame must have a 'Volume_BTC' column")

    subset = df[['High', 'Low', 'Close', volume_column]]
    return subset.iloc[::60].copy()
