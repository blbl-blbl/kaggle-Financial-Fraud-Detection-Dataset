# test_features.py

import numpy as np
import pandas as pd

from src.features import (
    feature_log_amount,
    time_features
)


def test_feature_log_amount():

    df = pd.DataFrame({
        "amount": [0, 100, 200]
    })

    result = feature_log_amount(df)

    expected = np.log1p([0, 100, 200])

    np.testing.assert_allclose(
        result["log_amount"],
        expected
    )


def test_time_features():

    df = pd.DataFrame({
        "step": [1, 7, 13, 19, 25]
    })
    result = time_features(df)

    assert result["day"].tolist() == [
        0, 0, 0, 0, 1
    ]

    expected_hour_sin = np.sin(
        2 * np.pi
        * np.array([0, 6, 12, 18, 0])
        / 24
    )

    expected_hour_cos = np.cos(
        2 * np.pi
        * np.array([0, 6, 12, 18, 0])
        / 24
    )

    np.testing.assert_allclose(
        result['hour_sin'],
        expected_hour_sin,
        atol=1e-10
    )

    np.testing.assert_allclose(
        result['hour_cos'],
        expected_hour_cos,
        atol=1e-10
    )