"""Tests for src.models.predict (Phase 6).

Covers the prediction path used by the Streamlit app:
- the complete default input row (no NaNs, survives the training pipeline)
- overlaying user inputs and input validation
- one-hot option parsing from the artifact's feature columns
- end-to-end prediction against the saved model artifact (skipped when the
  artifact is not present, e.g. in a fresh clone without local artifacts)
"""

import numpy as np
import pytest

from src.config import PATHS
from src.features.feature_engineering import engineer_features
from src.features.preprocessing import preprocess
from src.models import predict
from src.models.predict import (
    build_input_row,
    load_model_artifact,
    options_for,
    predict_price,
    prepare_user_row,
)

HAS_MODEL = PATHS.best_model.is_file()
requires_model = pytest.mark.skipif(not HAS_MODEL, reason="model artifact not present")


def test_module_importable():
    """The prediction module exposes its entry points."""
    assert callable(predict.predict_price)
    assert callable(predict.load_model_artifact)


def test_default_row_is_complete_and_pipeline_ready():
    """The default row must cover every raw column with no NaNs, and survive
    the exact training-time preprocessing + feature engineering."""
    row = predict._default_raw_row()
    assert row.shape[0] == 1
    assert row.isna().sum().sum() == 0, "default row contains NaNs"
    # Every raw feature column is present (Id/SalePrice excluded).
    assert "OverallQual" in row.columns
    assert "GrLivArea" in row.columns
    assert "Neighborhood" in row.columns
    assert "Id" not in row.columns
    assert "SalePrice" not in row.columns

    processed = engineer_features(preprocess(row.copy()))
    assert processed.isna().sum().sum() == 0, "pipeline introduced NaNs"
    assert "TotalSF" in processed.columns


def test_build_input_row_overlays_features():
    row = build_input_row(
        {"OverallQual": 9, "GrLivArea": 2500, "Neighborhood": "Gilbert"}
    )
    assert row.loc[0, "OverallQual"] == 9
    assert row.loc[0, "GrLivArea"] == 2500
    assert row.loc[0, "Neighborhood"] == "Gilbert"
    # Untouched features keep their default (data-derived or fallback).
    assert row.loc[0, "LotArea"] > 0


def test_build_input_row_skips_none_values():
    row = build_input_row({"OverallQual": None})
    assert row.loc[0, "OverallQual"] is not None


def test_build_input_row_rejects_unknown_feature():
    with pytest.raises(ValueError, match="Unknown feature"):
        build_input_row({"NotAFeature": 1})


def test_build_input_row_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        build_input_row({"GrLivArea": -100})


def test_build_input_row_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        build_input_row({"GrLivArea": float("nan")})


def test_options_for_parses_one_hot_levels():
    columns = [
        "Neighborhood_CollgCr",
        "Neighborhood_OldTown",
        "MSZoning_RL",
        "OverallQual",  # numeric column - must not match the prefix
    ]
    assert options_for(columns, "Neighborhood") == ["CollgCr", "OldTown"]
    # A column that was not one-hot encoded yields no levels.
    assert options_for(columns, "KitchenQual") == []


@requires_model
def test_predict_price_returns_finite_positive_price():
    price = predict_price(
        {"OverallQual": 7, "GrLivArea": 1800, "Neighborhood": "CollgCr"}
    )
    assert np.isfinite(price)
    assert price > 0
    # A sane Ames price band for a mid-range home.
    assert 10_000 < price < 2_000_000


@requires_model
def test_predict_price_matches_manual_pipeline():
    """predict_price must equal the manual pipeline: raw row -> preprocess ->
    engineer -> one-hot align -> model -> expm1 (guards against drift)."""
    artifact = load_model_artifact()
    features = {"OverallQual": 6, "GrLivArea": 1500, "Neighborhood": "Gilbert"}
    X = prepare_user_row(build_input_row(features), artifact["feature_columns"])
    expected = float(np.expm1(artifact["model"].predict(X)[0]))

    assert predict_price(features) == pytest.approx(expected)


@requires_model
def test_artifact_has_expected_shape():
    artifact = load_model_artifact()
    assert artifact["target_transform"] == "log1p"
    assert len(artifact["feature_columns"]) > 200
    # Every one-hot Neighborhood level the app offers exists in the model.
    for lvl in options_for(artifact["feature_columns"], "Neighborhood"):
        assert f"Neighborhood_{lvl}" in artifact["feature_columns"]
