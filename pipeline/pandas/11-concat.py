#!/usr/bin/env python3
"""Build a combined Coinbase/Bitstamp dataset with labeled origins."""

import pandas as pd

index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenate Coinbase and Bitstamp data with multi-level keys.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Coinbase data that includes a `Timestamp` column.
    df2 : pandas.DataFrame
        Bitstamp data that includes a `Timestamp` column.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame keyed by exchange name with Bitstamp rows whose
        timestamp is less than or equal to 1417411920 stacked above Coinbase.
    """
    cutoff_timestamp = 1417411920

    coinbase = index(df1.copy())
    bitstamp = index(df2.copy())

    bitstamp_subset = bitstamp[bitstamp.index <= cutoff_timestamp]

    return pd.concat(
        [bitstamp_subset, coinbase],
        keys=['bitstamp', 'coinbase']
    )
