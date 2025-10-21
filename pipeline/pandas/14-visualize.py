#!/usr/bin/env python3
"""Transform raw Coinbase minute data into cleaned daily aggregates."""

import pandas as pd


def visualize(df):
    """
    Clean and aggregate minute-resolution market data for visualization.

    Workflow:
        1. Validate required columns are present so downstream steps can assume
           their existence.
        2. Remove columns not needed for plotting (`Weighted_Price`).
        3. Standardize the timestamp by renaming it to `Date`, converting the
           UNIX seconds to pandas timestamps, and indexing on that date.
        4. Fill data gaps: forward-fill close prices, then reuse those values
           for open/high/low, and zero-fill missing volumes.
        5. Filter to entries from 2017 onward and resample the minute data to
           daily granularity using the specified aggregations.

    Parameters
    ----------
    df : pandas.DataFrame
        Input market data containing at least timestamp, price, and volume
        columns.

    Returns
    -------
    pandas.DataFrame
        Daily aggregated statistics with timestamps as the index.
    """
    required_price_columns = ['Close', 'High', 'Low', 'Open']
    required_volume_columns = ['Volume_(BTC)', 'Volume_(Currency)']

    for column in required_price_columns + required_volume_columns + ['Timestamp']:
        if column not in df.columns:
            raise KeyError(f"DataFrame must have a '{column}' column")

    transformed = df.copy()

    if 'Weighted_Price' in transformed.columns:
        transformed = transformed.drop(columns=['Weighted_Price'])

    # Normalize timestamps so every row belongs to its calendar day before
    # resampling.
    transformed = transformed.rename(columns={'Timestamp': 'Date'})
    transformed['Date'] = pd.to_datetime(transformed['Date'], unit='s').dt.normalize()
    transformed = transformed.set_index('Date')
    transformed = transformed.sort_index()

    # Propagate the last observed close price to fill missing values, ensuring
    # continuity for subsequent aggregations.
    transformed['Close'] = transformed['Close'].fillna(method='ffill')

    # With `Close` populated, reuse it to fill any remaining price gaps.
    for column in ['High', 'Low', 'Open']:
        transformed[column] = transformed[column].fillna(transformed['Close'])

    # Replace missing volume measurements with zeros prior to aggregation.
    for column in ['Volume_(BTC)', 'Volume_(Currency)']:
        transformed[column] = transformed[column].fillna(0)

    # Focus visualization on the requested 2017+ window.
    filtered = transformed.loc['2017-01-01':]

    # Aggregate the filtered minutes into daily values using the provided rules.
    daily = filtered.resample('D').agg({
        'High': 'max',
        'Low': 'min',
        'Open': 'mean',
        'Close': 'mean',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum'
    })

    return daily
