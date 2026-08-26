# test_threshold.py

import numpy as np
from src.threshold import find_threshold

def test_find_threshold_returns_valid_result():

    y_true = np.array([0, 0, 0, 1, 1])
    y_proba = np.array([0.05, 0.10, 0.20, 0.70, 0.90])

    result = find_threshold(
        y_true,
        y_proba,
        beta=2
    )

    assert 0 <= result["threshold"] <= 1
    assert 0 <= result["precision"] <= 1
    assert 0 <= result["recall"] <= 1
    assert 0 <= result["fbeta"] <= 1


def test_find_threshold_on_perfect_predictions():

    y_true = np.array([0, 0, 0, 1, 1])
    y_proba = np.array([0.01, 0.05, 0.10, 0.80, 0.90])

    result = find_threshold(
        y_true,
        y_proba,
        beta=2
    )

    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert np.isclose(result["fbeta"], 1.0)
