#!/usr/bin/env python3
import pandas as pd

# Create a dictionary where each key is a column name and each value is a list of column values
data = {
    "First": [0.0, 0.5, 1.0, 1.5],
    "Second": ["one", "two", "three", "four"]
}

# Create the DataFrame and specify the custom row labels (index)
df = pd.DataFrame(data, index= ["A", "B", "C", "D"])
