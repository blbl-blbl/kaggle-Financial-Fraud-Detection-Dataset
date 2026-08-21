# data.py

import pandas as pd
from pathlib import Path

from config import (
    BASE_FEATURES,
    TARGET
)

def load_data(path: Path):
    return pd.read_csv(path)

def validate_columns(df):
    required_columns = BASE_FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError (
            f"Missing columns are {missing_columns}"
        )

    return df[required_columns].copy()