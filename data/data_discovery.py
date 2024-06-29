import os

import pandas as pd

data = pd.read_csv(os.path.join("data", "synthetic_data.csv"))
print(data.describe())  # Check for strange values, min/max, and NaN counts
print(data.isna().sum())  # Check for NaN values
