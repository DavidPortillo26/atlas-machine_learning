#!/usr/bin/env python3
"""Combine exchange data with timestamp-first multi-index ordering."""

import pandas as pd

index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenate Bitstamp and Coinbase slices with timestamp leading the index.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Coinbase data containing a `Timestamp` column.
    df2 : pandas.DataFrame
        Bitstamp data containing a `Timestamp` column.

    Returns
    -------
    pandas.DataFrame
        Multi-indexed DataFrame whose first level is
        `Timestamp` (chronological)
        and the second level identifies the exchange
        (`bitstamp` or `coinbase`).
    """
    start_timestamp = 1417411980
    end_timestamp = 1417417980

    coinbase = index(df1.copy()).sort_index()
    bitstamp = index(df2.copy()).sort_index()

    coinbase_slice = coinbase.loc[start_timestamp:end_timestamp]
    bitstamp_slice = bitstamp.loc[start_timestamp:end_timestamp]

    combined = pd.concat(
        [bitstamp_slice, coinbase_slice],
        keys=['bitstamp', 'coinbase']
    )

    # Move timestamp to the leading level and sort for chronological display.
    combined = combined.swaplevel(0, 1)
    combined.index = combined.index.set_names(['Timestamp', None])
    return combined.sort_index(level=[0, 1])
