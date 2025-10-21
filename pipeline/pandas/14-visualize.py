#!/usr/bin/env python3
"""Transform market data for visualization."""

import pandas as pd


def visualize(df):
    """
    Prepare the dataset for daily visualization from 2017 onwards.

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

    transformed = transformed.rename(columns={'Timestamp': 'Date'})
    transformed['Date'] = pd.to_datetime(transformed['Date'], unit='s').dt.normalize()
    transformed = transformed.set_index('Date')
    transformed = transformed.sort_index()

    transformed['Close'] = transformed['Close'].fillna(method='ffill')

    for column in ['High', 'Low', 'Open']:
        transformed[column] = transformed[column].fillna(transformed['Close'])

    for column in ['Volume_(BTC)', 'Volume_(Currency)']:
        transformed[column] = transformed[column].fillna(0)

    filtered = transformed.loc['2017-01-01':]

    daily = filtered.resample('D').agg({
        'High': 'max',
        'Low': 'min',
        'Open': 'mean',
        'Close': 'mean',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum'
    })

    return daily
