"""Phase 2 pipeline: raw CSVs -> preprocessed -> feature-engineered -> saved.

Run from the project root:

    python -m src.pipeline

Outputs (gitignored):
    data/processed/train_processed.csv
    data/processed/test_processed.csv
"""

from __future__ import annotations

import pandas as pd

from src.config import PATHS
from src.data.load_data import load_raw_data
from src.features.feature_engineering import engineer_features
from src.features.preprocessing import preprocess


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full preparation pipeline on train and test.

    Returns
    -------
    (train, test) : tuple[pd.DataFrame, pd.DataFrame]
        The processed, feature-engineered DataFrames (also saved to
        ``data/processed/``).
    """
    train_raw, test_raw = load_raw_data()

    train = engineer_features(preprocess(train_raw))
    test = engineer_features(preprocess(test_raw))

    PATHS.processed_data_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(PATHS.train_processed, index=False)
    test.to_csv(PATHS.test_processed, index=False)

    return train, test


if __name__ == "__main__":
    train_df, test_df = prepare_data()
    print("train processed:", train_df.shape)
    print("test processed :", test_df.shape)
    print("saved to:", PATHS.processed_data_dir)
