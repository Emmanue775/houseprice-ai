"""Kaggle submission preparation (Phase 4).

Loads the best model saved in Phase 3 (``models/best_model.joblib``), applies
the **exact same** feature preparation used during training, predicts every
row of the real test set, converts back from log-space, validates the result
against Kaggle's format, and writes:

    data/processed/kaggle_submission.csv   (columns: Id, SalePrice)
    reports/submission_report.md           (validation report)

Run from the project root:

    python -m src.models.submission

Nothing here modifies the model or the training pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.config import ID_COLUMN, PATHS, PROJECT_ROOT
from src.data.load_data import load_processed_data
from src.models.train import prepare_features

SUBMISSION_PATH: Path = PATHS.processed_data_dir / "kaggle_submission.csv"
REPORT_PATH: Path = PROJECT_ROOT / "reports" / "submission_report.md"

EXPECTED_TEST_ROWS: int = 1459  # rows in the official test.csv


def prepare_features_for_prediction(
    test_df: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """One-hot encode a single frame and align it to the training columns.

    This mirrors the training-time transformation (``prepare_features`` in
    ``src/models/train.py``) for a standalone frame: object columns are
    one-hot encoded and the result is re-indexed to ``feature_columns``
    (unknown categories -> 0), so the model sees identical input.

    Parameters
    ----------
    test_df : pd.DataFrame
        Processed (preprocessed + engineered) data, e.g. a user row from the
        Streamlit app.
    feature_columns : list[str]
        The exact column order the model was trained on
        (``artifact["feature_columns"]``).

    Returns
    -------
    pd.DataFrame
        Model-ready feature matrix with columns == ``feature_columns``.
    """
    obj_cols = test_df.select_dtypes(include=["object", "category"]).columns.tolist()
    X = pd.get_dummies(test_df, columns=obj_cols, dtype=np.int8) if obj_cols else test_df.copy()
    return X.reindex(columns=feature_columns, fill_value=0)


def make_submission(
    model: object, feature_columns: list[str], test_df: pd.DataFrame
) -> pd.DataFrame:
    """Predict test prices with a fitted model and return an Id/SalePrice frame.

    The model predicts ``log1p(SalePrice)`` (trained that way in Phase 3), so
    predictions are converted back with ``np.expm1``.

    Parameters
    ----------
    model : fitted sklearn estimator
    feature_columns : list[str]
        Training column order.
    test_df : pd.DataFrame
        Processed test data, must contain the ``Id`` column.

    Returns
    -------
    pd.DataFrame
        Exactly two columns: ``Id`` (original order) and ``SalePrice``.
    """
    X_test = prepare_features_for_prediction(test_df, feature_columns)
    log_pred = model.predict(X_test)
    prices = np.expm1(log_pred)

    ids = test_df[ID_COLUMN] if ID_COLUMN in test_df.columns else np.arange(len(test_df))
    return pd.DataFrame({"Id": ids, "SalePrice": prices})


def validate_submission(
    submission: pd.DataFrame, expected_rows: Optional[int] = None
) -> dict:
    """Run every sanity check on the submission; raise on any violation.

    Checks (matching the Phase 4 requirements):
    - exactly the two columns ``Id``, ``SalePrice``
    - correct number of rows (if ``expected_rows`` given)
    - no missing values anywhere
    - no NaN / infinite / negative SalePrice values

    Returns a summary dict used for the validation report.
    """
    problems: list[str] = []

    if list(submission.columns) != ["Id", "SalePrice"]:
        problems.append(
            f"columns are {list(submission.columns)}, expected ['Id', 'SalePrice']"
        )
    if expected_rows is not None and len(submission) != expected_rows:
        problems.append(f"{len(submission)} rows, expected {expected_rows}")

    prices = submission["SalePrice"].to_numpy(dtype=float)
    missing = int(submission.isna().sum().sum())
    if missing:
        problems.append(f"{missing} missing values")

    if not np.isfinite(prices).all():
        problems.append("SalePrice contains NaN or infinite values")
    if not (prices > 0).all():
        problems.append("SalePrice contains non-positive values")

    if problems:
        raise ValueError("Submission failed validation: " + "; ".join(problems))

    return {
        "rows": int(len(submission)),
        "min_price": float(prices.min()),
        "max_price": float(prices.max()),
        "mean_price": float(prices.mean()),
        "missing": missing,
        "head": submission.head(5),
    }


def create_submission(
    submission_path: Path = SUBMISSION_PATH,
    report_path: Path = REPORT_PATH,
) -> dict:
    """Generate, validate, and save the final Kaggle submission.

    Returns the validation summary dict.
    """
    artifact = joblib.load(PATHS.best_model)
    train_df, test_df = load_processed_data()

    # Reuse the exact training-time feature preparation and verify the column
    # order matches the saved artifact (guards against any drift).
    _, X_test, _, feature_columns = prepare_features(train_df, test_df)
    if feature_columns != artifact["feature_columns"]:
        raise ValueError(
            "Feature columns differ between the saved artifact and the current "
            "pipeline - retrain or investigate before submitting."
        )

    submission = make_submission(artifact["model"], artifact["feature_columns"], test_df)
    summary = validate_submission(submission, expected_rows=EXPECTED_TEST_ROWS)

    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    _write_report(report_path, summary, submission_path)

    return summary


def _write_report(report_path: Path, summary: dict, submission_path: Path) -> None:
    head = summary["head"]
    # Note: iterrows() upcasts int columns to float, so format Id as :.0f
    head_rows = "\n".join(
        f"| {row['Id']:.0f} | {row['SalePrice']:.2f} |" for _, row in head.iterrows()
    )
    report = f"""# Kaggle submission report - HousePrice-AI (Phase 4)

- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
- Model: `models/best_model.joblib` (XGBoost, trained on `log1p(SalePrice)`)
- Submission file: `{submission_path}`

## Validation summary

| check | value |
|---|---|
| Rows | {summary['rows']} |
| Columns | Id, SalePrice |
| Minimum predicted price | ${summary['min_price']:,.0f} |
| Maximum predicted price | ${summary['max_price']:,.0f} |
| Mean predicted price | ${summary['mean_price']:,.0f} |
| Missing predictions | {summary['missing']} |
| NaN / infinite / negative values | none |

## First 5 rows

| Id | SalePrice |
|---|---|
{head_rows}

> Predictions come from the trained model only - no manual adjustment.
> Format: header `Id,SalePrice` with {summary['rows']} data rows (Kaggle-compatible).
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    summary = create_submission()
    print(f"Submission written to {SUBMISSION_PATH}")
    print(
        f"rows={summary['rows']} | min=${summary['min_price']:,.0f} | "
        f"max=${summary['max_price']:,.0f} | mean=${summary['mean_price']:,.0f} | "
        f"missing={summary['missing']}"
    )
