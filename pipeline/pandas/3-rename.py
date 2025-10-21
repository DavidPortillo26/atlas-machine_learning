#!/usr/bin/env python3
"""Rename the timestamp column and expose only the relevant fields."""

import pandas as pd


def rename(df):
    """
    Convert the `Timestamp` column to datetime and return selected columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing at least the `Timestamp` and `Close` columns.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the timestamp converted to datetime values and only
        the `Datetime` and `Close` columns retained.
    """
    if 'Timestamp' not in df.columns:
        raise KeyError("DataFrame must have a 'Timestamp' column")
    if 'Close' not in df.columns:
        raise KeyError("DataFrame must have a 'Close' column")

    # Work on a copy to avoid mutating the original DataFrame handed in.
    renamed = df.copy()
    renamed = renamed.rename(columns={'Timestamp': 'Datetime'})
    renamed['Datetime'] = pd.to_datetime(renamed['Datetime'], unit='s')
    return renamed.loc[:, ['Datetime', 'Close']]
