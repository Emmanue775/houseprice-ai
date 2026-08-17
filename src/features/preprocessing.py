"""Preprocessing: cleaning, missing values, and type fixes.

Phase 2 implementation notes (grounded in the official data dictionary and
verified on the actual files in ``data/raw/``):

- Many ``NA`` values are **not** missing data — they mean *"the feature does
  not exist"* (e.g. ``Alley`` = no alley access, ``BsmtQual`` = no basement,
  ``GarageType`` = no garage). Those are filled with the category ``"None"``
  instead of being imputed. Verified: rows with ``PoolQC``/``BsmtQual``/
  ``GarageType`` == NaN always have ``PoolArea``/``TotalBsmtSF``/``GarageArea``
  == 0.
- Ordinal ratings (``Po`` < ``Fa`` < ``TA`` < ``Gd`` < ``Ex``) are mapped to
  integers 1..5 so models see the natural ordering (0 = feature absent).
- ``MSSubClass`` and ``MoSold`` are stored as integers but are nominal codes,
  so they are cast to strings (one-hot encoding in Phase 3).
- ``LotFrontage`` is the main column with genuine missingness; it is imputed
  with the median ``LotFrontage`` of the same ``Neighborhood``.
- A final check guarantees no NaNs survive the pipeline.
"""

from __future__ import annotations

import pandas as pd

# Columns where NA/None is a legitimate category meaning "feature does not
# exist". After fillna("None") these are treated like any other category.
NO_FEATURE_CATEGORICAL = [
    "Alley",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "MasVnrType",  # "None" category + a few genuinely missing rows
]

# Numeric columns where NA means "no feature" -> fill with 0.
NO_FEATURE_NUMERIC = [
    "MasVnrArea",
    "GarageYrBlt",
    "GarageCars",
    "GarageArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
]

# Categorical columns with only a handful of missing rows -> fill with the
# most frequent value (the mode). (Modes verified on data/raw/train.csv.)
MODE_FILL = [
    "MSZoning",
    "Utilities",
    "Electrical",
    "KitchenQual",
    "Functional",
    "Exterior1st",
    "Exterior2nd",
    "SaleType",
]

# Nominal codes that pandas read as integers -> cast to string.
NOMINAL_INT = ["MSSubClass", "MoSold"]

# Ordinal mappings (higher = better). "None" is the "no feature" category
# produced by the fill step above, mapped to 0.
QUALITY_ORDINAL = {
    "ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "None": 0},
    "ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "None": 0},
    "BsmtQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "None": 0},
    "BsmtCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "None": 0},
    "HeatingQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "None": 0},
    "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "None": 0},
    "FireplaceQu": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "None": 0},
    "GarageQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "None": 0},
    "GarageCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "None": 0},
    "PoolQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "None": 0},
    # Basement exposure: No < Mn < Av < Gd
    "BsmtExposure": {"No": 1, "Mn": 2, "Av": 3, "Gd": 4, "None": 0},
    # Basement finish type: Unf < LwQ < Rec < BLQ < ALQ < GLQ
    "BsmtFinType1": {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6, "None": 0},
    "BsmtFinType2": {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6, "None": 0},
    # Home functionality: Sal < Sev < Maj2 < Maj1 < Mod < Min2 < Min1 < Typ
    "Functional": {
        "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4,
        "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8,
    },
    # Fence: no fence < wood/wire < good wood < min privacy < good privacy
    "Fence": {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
    # Land slope: severe < moderate < gentle
    "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
    # Lot shape: irregular < slightly irregular < regular
    "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
}

# Simple two/three-level maps.
BINARY_MAP = {"N": 0, "Y": 1}
PAVED_DRIVE = {"N": 0, "P": 1, "Y": 2}


def _fill_no_feature_categorical(df: pd.DataFrame) -> None:
    """Replace NaN with the "None" category where it means 'no feature'."""
    for col in NO_FEATURE_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].fillna("None")


def _fill_no_feature_numeric(df: pd.DataFrame) -> None:
    """Replace NaN with 0 where it means 'no feature' (e.g. no basement)."""
    for col in NO_FEATURE_NUMERIC:
        if col in df.columns:
            df[col] = df[col].fillna(0)


def _impute_lot_frontage(df: pd.DataFrame) -> None:
    """Impute LotFrontage with the median of the same Neighborhood.

    Falls back to the global median for neighborhoods that have no valid
    LotFrontage values at all (defensive; does not happen in the real data).
    """
    if "LotFrontage" not in df.columns:
        return
    median_by_neighborhood = df.groupby("Neighborhood")["LotFrontage"].transform("median")
    df["LotFrontage"] = df["LotFrontage"].fillna(median_by_neighborhood)
    df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())


def _fill_mode(df: pd.DataFrame) -> None:
    """Fill the few remaining missing categorical values with the mode."""
    for col in MODE_FILL:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])


def _apply_ordinals(df: pd.DataFrame) -> None:
    """Map ordinal quality ratings to integers 0..N."""
    for col, mapping in QUALITY_ORDINAL.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)
    for col, mapping in (("CentralAir", BINARY_MAP), ("PavedDrive", PAVED_DRIVE)):
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)


def _cast_nominal(df: pd.DataFrame) -> None:
    """Cast nominal integer codes (MSSubClass, MoSold) to strings."""
    for col in NOMINAL_INT:
        if col in df.columns:
            df[col] = df[col].astype(str)


def _assert_no_missing(df: pd.DataFrame) -> None:
    """Fail loudly if any missing values survived the pipeline."""
    remaining = df.columns[df.isna().any()].tolist()
    if remaining:
        raise ValueError(                f"Missing values remain after preprocessing: {remaining}. "
                "This must never happen - fix the preprocessing step."
        )


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full preprocessing pipeline to a raw DataFrame.

    The function is symmetric: the same steps are applied to train and test
    (each column is only touched if it exists in the frame).

    Parameters
    ----------
    df : pd.DataFrame
        Raw train or test data.

    Returns
    -------
    pd.DataFrame
        Cleaned data with no missing values, ready for feature engineering.

    Raises
    ------
    ValueError
        If any missing values remain after preprocessing.
    """
    df = df.copy()
    _fill_no_feature_categorical(df)
    _fill_no_feature_numeric(df)
    _impute_lot_frontage(df)
    _fill_mode(df)
    _apply_ordinals(df)
    _cast_nominal(df)
    _assert_no_missing(df)
    return df
