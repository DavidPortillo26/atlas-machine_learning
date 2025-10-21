#!/usr/bin/env python3
"""Set the timestamp column as the DataFrame index."""


def index(df):
    """
    Configure the DataFrame to use `Timestamp` as its index.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing a `Timestamp` column.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the `Timestamp` column promoted to index form.
    """
    if 'Timestamp' not in df.columns:
        raise KeyError("DataFrame must have a 'Timestamp' column")

    return df.set_index('Timestamp')
