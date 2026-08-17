"""Data loading utilities.

Responsibilities
----------------
- Read the raw Kaggle `train.csv` / `test.csv` files into pandas DataFrames.
- Validate that the expected files exist and raise helpful errors if not.

Phase 2 TODO: add a train/validation split helper so cross-validation never
leaks information between folds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from src.config import PATHS

PathLike = Union[str, Path]


def load_raw_data(
    train_path: PathLike = PATHS.train_raw,
    test_path: PathLike = PATHS.test_raw,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw Kaggle train and test CSVs.

    Parameters
    ----------
    train_path, test_path : str or Path
        Locations of the competition files. Defaults come from
        :data:`src.config.PATHS` (``data/raw/train.csv`` and
        ``data/raw/test.csv``).

    Returns
    -------
    (train_df, test_df) : tuple[pd.DataFrame, pd.DataFrame]
        The raw train and test DataFrames.

    Raises
    ------
    FileNotFoundError
        If either file is missing, with instructions on where to get it.
    """
    for label, path in (("train", train_path), ("test", test_path)):
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"{label}.csv not found at {path}. "
                "Download the official Kaggle 'House Prices' dataset and place "
                "train.csv / test.csv inside data/raw/ (see data/README.md)."
            )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def load_processed_data(
    train_path: PathLike = PATHS.train_processed,
    test_path: PathLike = PATHS.test_processed,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the processed (preprocessed + engineered) train/test CSVs.

    The processed files use the literal string ``"None"`` as a category that
    means "feature does not exist" (e.g. ``Alley`` = no alley access). Default
    ``pd.read_csv`` would re-parse that string as NaN, so we disable the
    default NA handling: after preprocessing there are *no* real missing
    values left, so nothing is lost.

    Returns
    -------
    (train_df, test_df) : tuple[pd.DataFrame, pd.DataFrame]

    Raises
    ------
    FileNotFoundError
        If the processed files are missing (run ``python -m src.pipeline``).
    """
    for label, path in (("train", train_path), ("test", test_path)):
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Processed {label}.csv not found at {path}. "
                "Run 'python -m src.pipeline' first."
            )

    train_df = pd.read_csv(train_path, keep_default_na=False)
    test_df = pd.read_csv(test_path, keep_default_na=False)
    return train_df, test_df
