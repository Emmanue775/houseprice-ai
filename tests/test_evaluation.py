"""Tests for src.evaluation.evaluate.

Verifies the RMSE helpers against hand-computed values using tiny arrays —
no dataset needed.
"""

import numpy as np
import pytest

from src.evaluation.evaluate import rmse, rmse_on_log


def test_rmse_known_values():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    expected = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    assert rmse(y_true, y_pred) == pytest.approx(expected)


def test_rmse_on_log_matches_rmsle_definition():
    y_true = np.array([100_000, 200_000, 350_000])
    y_pred = np.array([110_000, 180_000, 330_000])
    expected = float(
        np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
    )
    assert rmse_on_log(y_true, y_pred) == pytest.approx(expected)


def test_perfect_predictions_have_zero_error():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0
    assert rmse_on_log(y, y) == 0.0
