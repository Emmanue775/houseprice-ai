"""Tests for src.data.load_data.

These run without the Kaggle dataset: one test verifies the helpful error
raised when files are missing, the other verifies CSV reading with tiny
hand-made files.
"""

import pandas as pd
import pytest

from src.data.load_data import load_processed_data, load_raw_data


def test_load_raw_data_raises_when_files_missing(tmp_path):
    """Before the dataset is added, loading must fail with a clear error."""
    with pytest.raises(FileNotFoundError, match="train.csv"):
        load_raw_data(
            train_path=tmp_path / "train.csv",
            test_path=tmp_path / "test.csv",
        )


def test_load_raw_data_reads_csv(tmp_path):
    """Tiny CSVs should load into DataFrames with the expected shape."""
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"

    pd.DataFrame({"Id": [1, 2], "SalePrice": [100_000, 250_000]}).to_csv(
        train_file, index=False
    )
    pd.DataFrame({"Id": [3, 4], "LotArea": [5000, 7000]}).to_csv(
        test_file, index=False
    )

    train_df, test_df = load_raw_data(train_file, test_file)

    assert list(train_df.columns) == ["Id", "SalePrice"]
    assert train_df.shape == (2, 2)
    assert test_df.shape == (2, 2)


def test_load_processed_data_preserves_none_category(tmp_path):
    """The literal string 'None' must NOT be re-parsed as NaN."""
    train_file = tmp_path / "train_processed.csv"
    test_file = tmp_path / "test_processed.csv"

    pd.DataFrame({"Id": [1], "Alley": ["None"], "TotalSF": [2000]}).to_csv(
        train_file, index=False
    )
    pd.DataFrame({"Id": [2], "Alley": ["Pave"], "TotalSF": [1500]}).to_csv(
        test_file, index=False
    )

    train_df, test_df = load_processed_data(train_file, test_file)

    assert train_df.loc[0, "Alley"] == "None"
    assert not train_df.isna().any().any()
    assert test_df.loc[0, "Alley"] == "Pave"


def test_load_processed_data_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run 'python -m src.pipeline'"):
        load_processed_data(
            train_path=tmp_path / "train_processed.csv",
            test_path=tmp_path / "test_processed.csv",
        )
