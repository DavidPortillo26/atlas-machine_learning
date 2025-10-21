#!/usr/bin/env python3
"""Extract the last High/Close observations as a NumPy array."""


def array(df):
    """
    Return the last ten observations of the High and Close columns as an array.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing at least the `High` and `Close` columns.

    Returns
    -------
    numpy.ndarray
        A NumPy array with shape (10, 2) when at least ten rows are available.
        Fewer rows in the DataFrame will return as many as exist.
    """
    if 'High' not in df.columns:
        raise KeyError("DataFrame must have a 'High' column")
    if 'Close' not in df.columns:
        raise KeyError("DataFrame must have a 'Close' column")

    # Tail ensures we grab the most recent ten records in the expected order.
    return df[['High', 'Close']].tail(10).to_numpy()
