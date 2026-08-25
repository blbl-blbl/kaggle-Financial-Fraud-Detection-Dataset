# data.py

import pandas as pd
from src.config import (
    BASE_FEATURES,
    TARGET,
    PROJECT_PATH
)

def load_data(path=PROJECT_PATH / "data" / "Synthetic_Financial_datasets_log.csv"):
    df = pd.read_csv(path)
    df_checked = validate_columns(df)
    return df_checked

def validate_columns(df):
    required_columns = BASE_FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError (
            f"Missing columns are {missing_columns}"
        )

    return df[required_columns].copy()