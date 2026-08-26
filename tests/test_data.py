# test_data.py
from unittest import result

import pandas as pd
import pytest
from src.data import validate_columns
from src.config import BASE_FEATURES, TARGET


def test_validate_columns_raises_error():
    df = pd.DataFrame({
        "step": [1, 2],
        "amount": [100, 200]
    })

    with pytest.raises(ValueError):
        validate_columns(df)


def test_validate_columns_valid_dataframe():
    required_columns = BASE_FEATURES + [TARGET]

    df = pd.DataFrame({
        column: [0, 1] for column in required_columns
    })
    result = validate_columns(df)

    assert result.columns.tolist() == required_columns
    assert len(result) == 2