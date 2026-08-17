"""Tests for src.models.submission (Phase 4).

Uses tiny synthetic data and a stub model - no dataset needed.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.submission import (
    make_submission,
    prepare_features_for_prediction,
    validate_submission,
)


class _StubLogModel:
    """A stand-in for the trained model: predicts log(price) = f(sqft)."""

    def __init__(self, feature_columns):
        self.feature_columns = feature_columns

    def predict(self, X):
        sqft = X["GrLivArea"].to_numpy(dtype=float)
        return np.log1p(1000 + 100 * sqft)  # log1p of a positive price


def _test_frame(n=5, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Id": np.arange(1461, 1461 + n),
            "GrLivArea": rng.integers(500, 3000, n),
            "Neighborhood": rng.choice(["OldTown", "Gilbert"], n),
        }
    )


def _feature_columns(test_df):
    """Mimic training-time columns (one-hot on Neighborhood)."""
    return list(
        pd.get_dummies(test_df, columns=["Neighborhood"], dtype=np.int8).columns
    )


def test_prepare_features_alignment():
    test_df = _test_frame()
    cols = _feature_columns(test_df)

    X = prepare_features_for_prediction(test_df, cols)

    assert list(X.columns) == cols
    assert X.shape[0] == len(test_df)
    # No missing values introduced
    assert X.isna().sum().sum() == 0


def test_make_submission_columns_and_order():
    test_df = _test_frame()
    cols = _feature_columns(test_df)
    model = _StubLogModel(cols)

    sub = make_submission(model, cols, test_df)

    assert list(sub.columns) == ["Id", "SalePrice"]
    # Original IDs preserved in original order
    assert list(sub["Id"]) == list(test_df["Id"])
    assert len(sub) == len(test_df)


def test_make_submission_prices_are_positive():
    test_df = _test_frame()
    cols = _feature_columns(test_df)
    model = _StubLogModel(cols)

    sub = make_submission(model, cols, test_df)

    assert (sub["SalePrice"] > 0).all()
    assert np.isfinite(sub["SalePrice"]).all()


def test_validate_submission_passes_on_good_data():
    test_df = _test_frame()
    sub = make_submission(_StubLogModel(_feature_columns(test_df)), _feature_columns(test_df), test_df)

    summary = validate_submission(sub, expected_rows=len(test_df))

    assert summary["rows"] == len(test_df)
    assert summary["missing"] == 0
    assert summary["min_price"] > 0
    assert len(summary["head"]) == min(5, len(test_df))


def test_validate_submission_rejects_wrong_row_count():
    test_df = _test_frame(n=4)
    sub = make_submission(_StubLogModel(_feature_columns(test_df)), _feature_columns(test_df), test_df)

    with pytest.raises(ValueError, match="rows"):
        validate_submission(sub, expected_rows=1459)


def test_validate_submission_rejects_negative_prices():
    sub = pd.DataFrame({"Id": [1, 2], "SalePrice": [200_000.0, -5.0]})

    with pytest.raises(ValueError, match="non-positive"):
        validate_submission(sub)


def test_validate_submission_rejects_missing_values():
    sub = pd.DataFrame({"Id": [1, 2], "SalePrice": [200_000.0, np.nan]})

    with pytest.raises(ValueError, match="missing"):
        validate_submission(sub)
