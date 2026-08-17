"""Tests for src.features.feature_engineering.

Uses a tiny hand-made frame — no dataset needed.
"""

import pandas as pd

from src.features import feature_engineering
from src.features.feature_engineering import engineer_features


def _tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Id": [1, 2],
            "TotalBsmtSF": [800, 0],
            "1stFlrSF": [1000, 1200],
            "2ndFlrSF": [800, 0],
            "FullBath": [2, 1],
            "HalfBath": [1, 0],
            "BsmtFullBath": [1, 0],
            "BsmtHalfBath": [0, 0],
            "YrSold": [2010, 2008],
            "YearBuilt": [2000, 1960],
            "YearRemodAdd": [2005, 1960],
            "GarageYrBlt": [1999, 0],
            "GarageArea": [400, 0],
            "PoolArea": [0, 500],
            "Fireplaces": [1, 0],
            "OpenPorchSF": [50, 0],
            "EnclosedPorch": [0, 30],
            "3SsnPorch": [0, 0],
            "ScreenPorch": [20, 0],
            "WoodDeckSF": [100, 0],
            "OverallQual": [7, 4],
        }
    )


def test_total_sf():
    df = engineer_features(_tiny_frame())
    assert df.loc[0, "TotalSF"] == 800 + 1000 + 800  # 2600
    assert df.loc[1, "TotalSF"] == 0 + 1200 + 0  # 1200


def test_total_bath():
    df = engineer_features(_tiny_frame())
    assert df.loc[0, "TotalBath"] == 2 + 0.5 * 1 + 1 + 0.5 * 0  # 3.5
    assert df.loc[1, "TotalBath"] == 1.0


def test_house_age():
    df = engineer_features(_tiny_frame())
    assert df.loc[0, "HouseAge"] == 2010 - 2000  # 10
    assert df.loc[1, "HouseAge"] == 2008 - 1960  # 48


def test_garage_age_zero_when_no_garage():
    df = engineer_features(_tiny_frame())
    assert df.loc[0, "GarageAge"] == 2010 - 1999  # 11
    assert df.loc[1, "GarageAge"] == 0  # must not be 2008 - 0


def test_has_flags():
    df = engineer_features(_tiny_frame())
    assert df["HasGarage"].tolist() == [1, 0]
    assert df["HasPool"].tolist() == [0, 1]
    assert df["HasFireplace"].tolist() == [1, 0]
    assert df["HasBasement"].tolist() == [1, 0]
    assert df["Has2ndFloor"].tolist() == [1, 0]


def test_porch_and_outdoor_sf():
    df = engineer_features(_tiny_frame())
    assert df.loc[0, "TotalPorchSF"] == 50 + 0 + 0 + 20  # 70
    assert df.loc[0, "TotalOutdoorSF"] == 100 + 70  # 170
    assert df.loc[1, "TotalPorchSF"] == 0 + 30 + 0 + 0  # 30


def test_qual_sf_interaction():
    df = engineer_features(_tiny_frame())
    assert df.loc[0, "QualSF"] == 7 * 2600  # 18200


def test_module_importable():
    assert callable(feature_engineering.engineer_features)
