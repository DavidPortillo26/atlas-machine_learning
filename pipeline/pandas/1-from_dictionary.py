#!/usr/bin/env python3
"""
Module: 1-from_dictionary

Builds a pandas DataFrame from a
plain Python dictionary and exposes it as `df`.

Columns
-------
- "First": float values [0.0, 0.5, 1.0, 1.5]
- "Second": string values ["one", "two", "three", "four"]

Index
-----
["A", "B", "C", "D"]
"""

import pandas as pd

df = pd.DataFrame(
    {
        "First": [0.0, 0.5, 1.0, 1.5],
        "Second": ["one", "two", "three", "four"],
    },
    index=["A", "B", "C", "D"],
)
