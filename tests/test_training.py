"""Tests for src.models.train (Phase 3).

All tests use tiny synthetic data so they run in seconds without the
Kaggle dataset.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.models import train
from src.models.train import prepare_features, train_and_compare


def _synthetic(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Id": np.arange(1, n + 1),
            "OverallQual": rng.integers(1, 11, n),
            "GrLivArea": rng.integers(500, 3500, n),
            "Neighborhood": rng.choice(["OldTown", "Gilbert", "Sawyer"], n),
            "SalePrice": rng.uniform(50_000, 400_000, n),
        }
    )


def _fast_specs():
    """Tiny models so the end-to-end test finishes quickly."""
    return {
        "Ridge": {"estimator": lambda: Ridge(alpha=1.0)},
        "RandomForest": {
            "estimator": lambda: RandomForestRegressor(n_estimators=5, random_state=0)
        },
    }


def test_module_importable():
    assert callable(train.train_models)
    assert callable(train.train_and_compare)


def test_prepare_features_one_hot_and_log_target():
    train_df = _synthetic()
    test_df = _synthetic(n=10, seed=1)

    X_train, X_test, y, feature_columns = prepare_features(train_df, test_df)

    # Id and SalePrice must be gone; object column one-hot encoded
    assert "Id" not in X_train.columns
    assert "SalePrice" not in X_train.columns
    assert any(c.startswith("Neighborhood_") for c in X_train.columns)

    # Train and test share identical columns, test has no missing dummy cols
    assert list(X_train.columns) == feature_columns
    assert list(X_test.columns) == feature_columns

    # y is log1p of the target
    np.testing.assert_allclose(y, np.log1p(train_df["SalePrice"].to_numpy()))


def test_cross_validate_returns_fold_scores():
    train_df = _synthetic()
    X_train, _, y, _ = prepare_features(train_df, train_df)

    result = train.cross_validate_and_report(Ridge(alpha=1.0), X_train, y, n_splits=3)

    assert len(result["scores"]) == 3
    assert all(np.isfinite(s) and s > 0 for s in result["scores"])
    assert result["mean"] > 0


def test_train_and_compare_saves_artifacts(tmp_path):
    train_df = _synthetic()
    test_df = _synthetic(n=8, seed=2)
    X_train, X_test, y, feature_columns = prepare_features(train_df, test_df)

    out = train_and_compare(
        X_train,
        X_test,
        y,
        feature_columns,
        test_ids=test_df["Id"].to_numpy(),
        save_dir=tmp_path,
        specs=_fast_specs(),
        n_splits=3,
    )

    # Results surface
    assert out["best_model"] in _fast_specs()
    assert out["summary"].shape[0] == 2

    # Artifacts exist and are loadable
    assert (tmp_path / "models" / "best_model.joblib").is_file()
    assert (tmp_path / "reports" / "cv_results.csv").is_file()
    assert (tmp_path / "reports" / "model_comparison.md").is_file()
    assert (tmp_path / "reports" / "figures" / "model_comparison.png").is_file()

    import joblib

    artifact = joblib.load(tmp_path / "models" / "best_model.joblib")
    assert artifact["feature_columns"] == feature_columns
    assert artifact["target_transform"] == "log1p"

    # Predictions: one row per test row, real Ids, positive prices
    preds = pd.read_csv(tmp_path / "data" / "processed" / "test_predictions.csv")
    assert len(preds) == len(test_df)
    assert list(preds["Id"]) == list(test_df["Id"])
    assert (preds["SalePrice"] > 0).all()
