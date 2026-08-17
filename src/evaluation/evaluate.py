"""Evaluation metrics for the Kaggle House Prices task.

The competition is scored on root mean squared error (RMSE) between the
logarithm of the predicted value and the logarithm of the observed
``SalePrice``. We therefore report both plain RMSE and RMSE-on-log.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, cross_validate

from src.config import RANDOM_STATE

ArrayLike = Union[Sequence[float], np.ndarray]


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error between true and predicted values.

    ``sqrt(mean((y_true - y_pred) ** 2))``
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def rmse_on_log(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """RMSE computed on log1p-transformed values (the Kaggle metric).

    ``sqrt(mean((log1p(y_true) - log1p(y_pred)) ** 2))`` — equivalent to RMSLE.
    """
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))


def cross_validate_and_report(estimator, X, y, n_splits: int = 5) -> dict:
    """Cross-validate an estimator and return per-fold log-RMSE scores.

    The target ``y`` is expected to be **already log1p-transformed** (this
    project trains on ``log1p(SalePrice)``), so plain RMSE on this scale is
    exactly Kaggle's metric: RMSE between log1p(prediction) and log1p(truth).

    Parameters
    ----------
    estimator : sklearn estimator
        A fresh (unfitted) estimator or pipeline.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        log1p-transformed target.
    n_splits : int
        Number of KFold splits (shuffled, seeded via ``RANDOM_STATE``).

    Returns
    -------
    dict
        ``{"scores": [...], "mean": float, "std": float}`` where each score
        is the log-RMSE of one fold.
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scorer = make_scorer(rmse, greater_is_better=False)
    out = cross_validate(estimator, X, y, cv=cv, scoring=scorer)
    scores = list(-out["test_score"])  # sklearn returns negated scores
    return {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }
