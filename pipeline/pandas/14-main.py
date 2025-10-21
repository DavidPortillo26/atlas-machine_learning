#!/usr/bin/env python3
"""Visualize Coinbase daily OHLC and volume aggregates from 2017 onward."""

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file
visualize = __import__('14-visualize').visualize

# Load the min-by-min Coinbase data and reshape it into daily aggregates that
# have already been cleaned and gap-filled by `visualize`.
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
daily = visualize(df)

# Plot the daily open/high/low/close prices to observe overall trends.
ax = daily[['Open', 'High', 'Low', 'Close']].plot(
    figsize=(12, 6),
    title='Coinbase Daily OHLC (2017+)'
)
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')

# Plot daily traded volume (BTC and USD) to highlight periods of elevated
# activity.
fig2, ax2 = plt.subplots(figsize=(12, 4))
daily[['Volume_(BTC)', 'Volume_(Currency)']].plot(ax=ax2, title='Daily Volume (2017+)')
ax2.set_xlabel('Date')
ax2.set_ylabel('Volume')

plt.tight_layout()
plt.show()
