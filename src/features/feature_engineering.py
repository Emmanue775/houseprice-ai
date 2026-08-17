"""Feature engineering: domain-driven features derived from raw columns.

The features below are standard for the Ames Housing dataset and were chosen
after EDA (see ``notebooks/01_eda.ipynb``):

- Size: ``TotalSF``, ``TotalBath``, ``TotalPorchSF``, ``TotalOutdoorSF``
- Age: ``HouseAge``, ``RemodAge``, ``GarageAge`` (0 when no garage)
- Flags: ``HasGarage``, ``HasPool``, ``HasFireplace``, ``HasBasement``,
  ``Has2ndFloor``
- Interaction: ``QualSF`` = ``OverallQual`` * ``TotalSF`` (quality-weighted size)
"""

from __future__ import annotations

import pandas as pd


def _columns_exist(df: pd.DataFrame, cols: list[str]) -> bool:
    """Return True only if every column is present (avoids KeyErrors on partial frames)."""
    return all(col in df.columns for col in cols)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to a preprocessed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed train or test data (no missing values).

    Returns
    -------
    pd.DataFrame
        The same data with new feature columns added.
    """
    df = df.copy()

    if _columns_exist(df, ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]):
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    if _columns_exist(df, ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]):
        df["TotalBath"] = (
            df["FullBath"]
            + 0.5 * df["HalfBath"]
            + df["BsmtFullBath"]
            + 0.5 * df["BsmtHalfBath"]
        )

    if _columns_exist(df, ["YrSold", "YearBuilt"]):
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    if _columns_exist(df, ["YrSold", "YearRemodAdd"]):
        df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    if _columns_exist(df, ["YrSold", "GarageYrBlt"]):
        # GarageYrBlt == 0 means "no garage" -> GarageAge 0, not a huge number.
        df["GarageAge"] = (df["YrSold"] - df["GarageYrBlt"]).where(
            df["GarageYrBlt"] > 0, 0
        )

    if "GarageArea" in df.columns:
        df["HasGarage"] = (df["GarageArea"] > 0).astype(int)

    if "PoolArea" in df.columns:
        df["HasPool"] = (df["PoolArea"] > 0).astype(int)

    if "Fireplaces" in df.columns:
        df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)

    if "TotalBsmtSF" in df.columns:
        df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)

    if "2ndFlrSF" in df.columns:
        df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)

    if _columns_exist(df, ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]):
        df["TotalPorchSF"] = (
            df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
        )

    if "WoodDeckSF" in df.columns and "TotalPorchSF" in df.columns:
        df["TotalOutdoorSF"] = df["WoodDeckSF"] + df["TotalPorchSF"]

    if _columns_exist(df, ["OverallQual", "TotalSF"]):
        df["QualSF"] = df["OverallQual"] * df["TotalSF"]

    return df
