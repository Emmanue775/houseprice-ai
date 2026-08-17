"""Tests for src.features.preprocessing.

Uses a tiny hand-made Ames-like DataFrame — no dataset needed.
"""

import numpy as np
import pandas as pd

from src.features import preprocessing
from src.features.preprocessing import preprocess


def _tiny_ames_like() -> pd.DataFrame:
    """A minimal frame exercising every preprocessing branch."""
    return pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "MSSubClass": [60, 20, 60],
            "MoSold": [5, 7, 5],
            "Alley": ["Pave", np.nan, "Grvl"],  # NaN -> "None" category
            "Neighborhood": ["OldTown", "OldTown", "Gilbert"],
            "LotFrontage": [80.0, np.nan, np.nan],
            "MasVnrArea": [112.0, np.nan, 0.0],
            "BsmtQual": ["Gd", np.nan, "TA"],  # NaN -> "None" -> 0
            "ExterQual": ["Ex", "TA", "Fa"],  # 5, 3, 2
            "CentralAir": ["Y", "N", "Y"],  # 1, 0, 1
            "Electrical": ["SBrkr", np.nan, "SBrkr"],  # mode fill
            "GarageCars": [2, np.nan, 1],  # NaN -> 0
        }
    )


def test_no_feature_categories_become_none():
    df = preprocess(_tiny_ames_like())
    assert df.loc[1, "Alley"] == "None"  # kept as a category
    assert df.loc[0, "Alley"] == "Pave"


def test_ordinal_quality_mapping():
    df = preprocess(_tiny_ames_like())
    assert df.loc[0, "ExterQual"] == 5  # Ex
    assert df.loc[1, "ExterQual"] == 3  # TA
    assert df.loc[2, "ExterQual"] == 2  # Fa
    assert df.loc[1, "BsmtQual"] == 0  # no basement


def test_binary_mapping():
    df = preprocess(_tiny_ames_like())
    assert df["CentralAir"].tolist() == [1, 0, 1]


def test_nominal_codes_cast_to_string():
    df = preprocess(_tiny_ames_like())
    assert df["MSSubClass"].dtype == object
    assert df.loc[0, "MSSubClass"] == "60"
    assert df.loc[1, "MSSubClass"] == "20"


def test_numeric_no_feature_filled_with_zero():
    df = preprocess(_tiny_ames_like())
    assert df.loc[1, "MasVnrArea"] == 0.0
    assert df.loc[1, "GarageCars"] == 0


def test_lot_frontage_imputed_by_neighborhood_median():
    df = preprocess(_tiny_ames_like())
    # OldTown: median of [80] = 80; Gilbert has no median -> falls back to
    # global median. All values must be filled and positive.
    assert not df["LotFrontage"].isna().any()
    assert df.loc[1, "LotFrontage"] == 80.0


def test_mode_fill():
    df = preprocess(_tiny_ames_like())
    assert df.loc[1, "Electrical"] == "SBrkr"


def test_no_missing_values_survive():
    df = preprocess(_tiny_ames_like())
    assert df.isna().sum().sum() == 0


def test_module_importable():
    assert callable(preprocessing.preprocess)
