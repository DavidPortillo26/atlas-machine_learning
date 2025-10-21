#!/usr/bin/env python3
"""Clean price and volume data by filling missing values."""


def fill(df):
    """
    Prepare the DataFrame by dropping weighted price and filling gaps.

    Parameters
    ----------
    df : pandas.DataFrame
        Input market data containing price and volume columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame without the weighted price column and with missing values
        addressed according to the specification.
    """
    required_price_columns = ['Close', 'High', 'Low', 'Open']
    required_volume_columns = ['Volume_(BTC)', 'Volume_(Currency)']

    for column in required_price_columns + required_volume_columns:
        if column not in df.columns:
            raise KeyError(f"DataFrame must have a '{column}' column")

    cleaned = df.copy()

    if 'Weighted_Price' in cleaned.columns:
        cleaned = cleaned.drop(columns=['Weighted_Price'])

    cleaned['Close'] = cleaned['Close'].fillna(method='ffill')

    for column in ['High', 'Low', 'Open']:
        cleaned[column] = cleaned[column].fillna(cleaned['Close'])

    for column in ['Volume_(BTC)', 'Volume_(Currency)']:
        cleaned[column] = cleaned[column].fillna(0)

    return cleaned
