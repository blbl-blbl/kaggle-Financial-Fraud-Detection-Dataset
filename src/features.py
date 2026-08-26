# features.py

import pandas as pd
import numpy as np
from src.config import (
    FINAL_FEATURES,
    TARGET,
    CATEGORICAL_FEATURES
)

def build_features(df) -> pd.DataFrame:
    df_copy = df.copy()

    df_copy = feature_log_amount(df_copy)
    df_copy = time_features(df_copy)
    df_copy = feature_to_categorical(df_copy)

    required_columns = FINAL_FEATURES.copy()

    if TARGET in df_copy.columns:
        required_columns.append(TARGET)

    return df_copy[required_columns]


def feature_log_amount(df) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['log_amount'] = np.log1p(df_copy['amount'])
    return df_copy

def time_features(df) -> pd.DataFrame:
    df_copy = df.copy()
    hour = (df_copy["step"] - 1) % 24
    df_copy['day'] = (df_copy['step'] - 1) // 24
    df_copy['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df_copy['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    return df_copy

def feature_to_categorical(
        df,
        cat_cols=CATEGORICAL_FEATURES
) -> pd.DataFrame:

    df_copy = df.copy()
    df_copy[cat_cols] = df_copy[cat_cols].astype('category')
    return df_copy