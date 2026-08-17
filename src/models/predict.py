"""Model inference for the Streamlit app (Phase 6).

``predict_price`` turns a dict of raw Kaggle-column inputs (e.g.
``{"OverallQual": 8, "GrLivArea": 1800, "Neighborhood": "CollgCr"}``) into a
sale-price estimate using the **exact same transforms used during training**:

    raw user row
        -> preprocess()              (src.features.preprocessing)
        -> engineer_features()       (src.features.feature_engineering)
        -> one-hot encode + align    (prepare_features_for_prediction)
        -> saved XGBoost model       (models/best_model.joblib)
        -> np.expm1()                (undo the log1p training target)
        -> predicted SalePrice

No second, parallel preprocessing pipeline lives here — the training-time
functions are imported and reused directly.

The pipeline is validated end-to-end by ``tests/test_prediction.py``.
"""

from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.config import ID_COLUMN, PATHS, TARGET
from src.features.feature_engineering import engineer_features
from src.features.preprocessing import NO_FEATURE_CATEGORICAL, NO_FEATURE_NUMERIC, preprocess
from src.models.submission import prepare_features_for_prediction

# ---------------------------------------------------------------------------
# Model artifact
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def load_model_artifact() -> dict:
    """Load the saved model artifact (cached after the first call).

    Returns
    -------
    dict
        ``{"model": fitted estimator, "feature_columns": list[str],
        "target_transform": str}`` as saved by ``src.models.train``.

    Raises
    ------
    FileNotFoundError
        If ``models/best_model.joblib`` does not exist (run
        ``python -m src.models.train`` first).
    """
    if not PATHS.best_model.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {PATHS.best_model}. "
            "Run 'python -m src.models.train' first."
        )
    return joblib.load(PATHS.best_model)


def options_for(feature_columns: List[str], column: str) -> List[str]:
    """Return the sorted category levels of a one-hot encoded column.

    The training pipeline one-hot encodes every object column, so the levels
    of ``Neighborhood`` live in the artifact's ``feature_columns`` as
    ``Neighborhood_<level>``. Parsing them back gives the exact vocabulary the
    model was trained on — no hard-coded category lists.

    Returns an empty list for columns that were not one-hot encoded (e.g.
    ordinal columns such as ``KitchenQual``, which preprocessing maps to
    integers).
    """
    prefix = column + "_"
    return sorted(c[len(prefix):] for c in feature_columns if c.startswith(prefix))


# ---------------------------------------------------------------------------
# Building a complete raw input row
# ---------------------------------------------------------------------------

# Static defaults used ONLY when data/raw/train.csv is unavailable at runtime
# (a fresh clone without the gitignored dataset). Values are typical Ames
# modal/median figures. When the raw training data is present (the normal
# case), defaults are derived from it instead, so unspecified features behave
# like an average house from the real training set.
_FALLBACK_DEFAULTS: Dict[str, Any] = {
    "MSSubClass": 20, "MSZoning": "RL", "LotFrontage": 70.0, "LotArea": 9478,
    "Street": "Pave", "Alley": "None", "LotShape": "Reg", "LandContour": "Lvl",
    "Utilities": "AllPub", "LotConfig": "Inside", "LandSlope": "Gtl",
    "Neighborhood": "CollgCr", "Condition1": "Norm", "Condition2": "Norm",
    "BldgType": "1Fam", "HouseStyle": "1Story", "OverallQual": 5, "OverallCond": 5,
    "YearBuilt": 1973, "YearRemodAdd": 1978, "RoofStyle": "Gable",
    "RoofMatl": "CompShg", "Exterior1st": "VinylSd", "Exterior2nd": "VinylSd",
    "MasVnrType": "None", "MasVnrArea": 0, "ExterQual": "TA", "ExterCond": "TA",
    "Foundation": "PConc", "BsmtQual": "TA", "BsmtCond": "TA",
    "BsmtExposure": "No", "BsmtFinType1": "Unf", "BsmtFinSF1": 0,
    "BsmtFinType2": "Unf", "BsmtFinSF2": 0, "BsmtUnfSF": 0, "TotalBsmtSF": 0,
    "Heating": "GasA", "HeatingQC": "TA", "CentralAir": "Y", "Electrical": "SBrkr",
    "1stFlrSF": 1162, "2ndFlrSF": 0, "LowQualFinSF": 0, "GrLivArea": 1464,
    "BsmtFullBath": 0, "BsmtHalfBath": 0, "FullBath": 2, "HalfBath": 0,
    "BedroomAbvGr": 3, "KitchenAbvGr": 1, "KitchenQual": "TA", "TotRmsAbvGrd": 6,
    "Functional": "Typ", "Fireplaces": 0, "FireplaceQu": "None",
    "GarageType": "Attchd", "GarageYrBlt": 0, "GarageFinish": "Unf",
    "GarageCars": 2, "GarageArea": 480, "GarageQual": "TA", "GarageCond": "TA",
    "PavedDrive": "Y", "WoodDeckSF": 0, "OpenPorchSF": 0, "EnclosedPorch": 0,
    "3SsnPorch": 0, "ScreenPorch": 0, "PoolArea": 0, "PoolQC": "None",
    "Fence": "None", "MiscFeature": "None", "MiscVal": 0, "MoSold": 6,
    "YrSold": 2008, "SaleType": "WD", "SaleCondition": "Normal",
}


@functools.lru_cache(maxsize=1)
def _defaults_from_data() -> Optional[Dict[str, Any]]:
    """Derive one default value per raw column from data/raw/train.csv.

    Uses the same semantics as the training preprocessing: "feature does not
    exist" columns default to ``"None"`` / ``0``, categoricals to their mode,
    numerics to their median. Returns ``None`` if the raw data is missing.
    """
    if not PATHS.train_raw.is_file():
        return None
    raw = pd.read_csv(PATHS.train_raw)
    defaults: Dict[str, Any] = {}
    for col in raw.columns:
        if col in (TARGET, ID_COLUMN):
            continue
        if col in NO_FEATURE_CATEGORICAL:
            defaults[col] = "None"
        elif col in NO_FEATURE_NUMERIC:
            defaults[col] = 0
        elif raw[col].dtype == object:
            mode = raw[col].mode(dropna=True)
            defaults[col] = str(mode.iloc[0]) if len(mode) else "None"
        else:
            median = raw[col].median(skipna=True)
            defaults[col] = 0.0 if pd.isna(median) else float(median)
    return defaults


def _default_raw_row() -> pd.DataFrame:
    """A single complete raw row (every feature column, no NaNs).

    Columns are the raw Kaggle feature columns (``Id`` / ``SalePrice``
    excluded). This guarantees ``preprocess()`` never trips its
    "no missing values" assertion on a one-row frame.
    """
    defaults = _defaults_from_data() or _FALLBACK_DEFAULTS
    return pd.DataFrame([defaults])


def build_input_row(features: Dict[str, Any]) -> pd.DataFrame:
    """Overlay user inputs onto the default raw row.

    Only known raw column names are accepted; ``None``/``NaN`` values are
    skipped so the default stands. Numeric inputs must be finite and
    non-negative (every Ames feature is naturally non-negative).

    Parameters
    ----------
    features : dict
        Raw-column -> value pairs, e.g. ``{"OverallQual": 8, "GrLivArea": 1800}``.

    Returns
    -------
    pd.DataFrame
        One-row DataFrame with every raw feature column present.

    Raises
    ------
    ValueError
        If a key is not a raw feature column or a numeric value is invalid.
    """
    row = _default_raw_row()
    for key, value in features.items():
        if key not in row.columns:
            raise ValueError(
                f"Unknown feature '{key}'. Expected raw Kaggle column names, "
                f"e.g. OverallQual, GrLivArea, Neighborhood."
            )
        if value is None:
            continue
        if isinstance(value, (int, float)) and not np.isfinite(value):
            raise ValueError(f"'{key}' must be a finite number, got {value!r}.")
        if isinstance(value, (int, float)) and value < 0:
            raise ValueError(f"'{key}' must be non-negative, got {value!r}.")
        row.loc[0, key] = value
    return row


def prepare_user_row(raw_row: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Run a raw row through the exact training-time transform.

    ``preprocess`` -> ``engineer_features`` -> one-hot encode & align to the
    artifact's ``feature_columns`` (unknown categories fall back to all-zero,
    matching the training-time ``prepare_features`` behaviour).
    """
    processed = engineer_features(preprocess(raw_row))
    return prepare_features_for_prediction(processed, feature_columns)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_price(features: Dict[str, Any]) -> float:
    """Predict a sale price (in dollars) from raw house characteristics.

    Parameters
    ----------
    features : dict
        Raw Kaggle column -> value pairs (see :func:`build_input_row`).

    Returns
    -------
    float
        Predicted ``SalePrice`` in dollars. The model predicts on the
        ``log1p(SalePrice)`` scale, so the result is ``np.expm1(log_pred)``.

    Raises
    ------
    ValueError
        On unknown feature names or invalid (negative / non-finite) values.
    FileNotFoundError
        If no trained model artifact exists.
    """
    artifact = load_model_artifact()
    X = prepare_user_row(build_input_row(features), artifact["feature_columns"])
    log_pred = float(artifact["model"].predict(X)[0])
    return float(np.expm1(log_pred))
