#!/usr/bin/env python3
"""
Module: 1-from_dictionary

Builds a pandas DataFrame from a plain Python dictionary and exposes it as
the variable `df`.

Data
----
Columns:
    - "First": float values [0.0, 0.5, 1.0, 1.5]
    - "Second": string values ["one", "two", "three", "four"]

Index:
    - Row labels: ["A", "B", "C", "D"]

Notes
-----
This module intentionally defines a top-level variable `df` (not a function),
as required by the exercise and by 1-main.py which imports `df`.
"""

import pandas as pd

# Column data as a plain Python dictionary. Keys become column names.
_data = {
    "First": [0.0, 0.5, 1.0, 1.5],
    "Second": ["one", "two", "three", "four"],
}

# Row labels (index) in the required order.
_index = ["A", "B", "C", "D"]

# The DataFrame expected by 1-main.py
df = pd.DataFrame(_data, index=_index)

# Optional: explicitly export only df if your tooling inspects __all__.
__all__ = ["df"]
