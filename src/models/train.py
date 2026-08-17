"""Model training: cross-validated comparison of regression models.

Phase 3 implementation. Run from the project root:

    python -m src.models.train

What happens
------------
1. Load the processed data produced by ``python -m src.pipeline``.
2. One-hot encode nominal categories (fitted on train only; test is aligned
   to the same columns, so nothing leaks).
3. Cross-validate every candidate model (5-fold, shuffled, seeded) on the
   ``log1p(SalePrice)`` target. Plain RMSE on that scale is exactly Kaggle's
   log-RMSE metric.
4. Pick the model with the lowest mean CV log-RMSE, run a small grid search
   on it, refit on all training data, and save reproducible artifacts:
     models/best_model.joblib            (model + feature_columns)
     reports/cv_results.csv              (per-fold log-RMSE, long format)
     reports/model_comparison.md         (human-readable report)
     reports/figures/model_comparison.png
     data/processed/test_predictions.csv (Id, SalePrice)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe: never opens a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ID_COLUMN, PATHS, PROJECT_ROOT, RANDOM_STATE, TARGET
from src.data.load_data import load_processed_data
from src.evaluation.evaluate import cross_validate_and_report

# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------


def prepare_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    """Build the model-ready feature matrices from processed DataFrames.

    - Drops the identifier and the target column.
    - One-hot encodes nominal (object) columns using ``pd.get_dummies``.
    - The dummies are fitted on **train only**; ``test`` is re-indexed to the
      train columns (fill 0) so train and test share identical columns.
    - Returns ``y`` = ``log1p(SalePrice)`` (the trained target).

    Returns
    -------
    (X_train, X_test, y, feature_columns)
    """
    feature_cols = [c for c in train_df.columns if c not in (TARGET, ID_COLUMN)]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    obj_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        X_train = pd.get_dummies(X_train, columns=obj_cols, dtype=np.int8)
        X_test = pd.get_dummies(X_test, columns=obj_cols, dtype=np.int8)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    y = np.log1p(train_df[TARGET].to_numpy(dtype=float))
    return X_train, X_test, y, X_train.columns.tolist()


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------


def _make_xgboost() -> Optional[object]:
    """Build the XGBoost regressor, or None if xgboost is not installed."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None
    return XGBRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )


def build_model_specs() -> Dict[str, Dict[str, Callable]]:
    """Return {name: {"estimator": factory}} for every candidate model.

    Factories are called later so every cross-validation fold gets a fresh,
    unfitted estimator (no leakage between folds).
    """
    specs = {
        "DummyBaseline": {
            "estimator": lambda: DummyRegressor(strategy="median"),
        },
        "Ridge": {
            "estimator": lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", Ridge(alpha=10.0)),
                ]
            ),
        },
        "ElasticNet": {
            "estimator": lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000)),
                ]
            ),
        },
        "RandomForest": {
            "estimator": lambda: RandomForestRegressor(
                n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
            ),
        },
        "GradientBoosting": {
            "estimator": lambda: GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
        },
        "XGBoost": {"estimator": _make_xgboost},
    }
    # Drop models whose (optional) dependency is missing.
    return {name: spec for name, spec in specs.items() if spec["estimator"]() is not None}


# Small tuning grids for the most promising families (applied only to the
# winner of the initial comparison, if it has a grid here).
TUNE_GRIDS: Dict[str, dict] = {
    "GradientBoosting": {
        "n_estimators": [150, 300],
        "learning_rate": [0.03, 0.05],
        "max_depth": [3, 4],
    },
    "XGBoost": {
        "n_estimators": [150, 300],
        "learning_rate": [0.03, 0.05],
        "max_depth": [3, 5],
    },
}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def train_and_compare(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y: np.ndarray,
    feature_columns: list[str],
    *,
    test_ids: Optional[np.ndarray] = None,
    save_dir: Optional[Path] = None,
    specs: Optional[Dict[str, Dict[str, Callable]]] = None,
    n_splits: int = 5,
) -> dict:
    """Cross-validate all models, pick the best, optionally save artifacts.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Feature matrices from :func:`prepare_features`.
    y : np.ndarray
        log1p-transformed target.
    feature_columns : list[str]
        Column order used to build the feature matrices.
    test_ids : np.ndarray or None
        Original test row identifiers (``Id``), used for the submission file.
    save_dir : Path or None
        If given, artifacts are written under this directory (test-friendly:
        tests pass a tmp dir). If None, nothing is saved.
    specs : dict or None
        Model catalogue; defaults to :func:`build_model_specs`.
    n_splits : int
        Number of CV folds.

    Returns
    -------
    dict
        Summary table, best model name, per-fold scores, tuned info.
    """
    specs = specs or build_model_specs()

    results: Dict[str, dict] = {}
    for name, spec in specs.items():
        estimator = spec["estimator"]()
        res = cross_validate_and_report(estimator, X_train, y, n_splits=n_splits)
        results[name] = res
        print(
            f"  {name:<16} log-RMSE {res['mean']:.4f} +/- {res['std']:.4f}"
        )

    summary = pd.DataFrame(
        {
            name: {"mean_log_rmse": r["mean"], "std_log_rmse": r["std"]}
            for name, r in results.items()
        }
    ).T.sort_values("mean_log_rmse")

    best_name = summary.index[0]
    print(f"\nBest model (default config): {best_name}")

    # --- Tune the winner with a small grid search --------------------------
    tuned: Optional[dict] = None
    grid = TUNE_GRIDS.get(best_name)
    if grid is not None:
        print(f"Tuning {best_name} with grid search...")
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        search = GridSearchCV(
            specs[best_name]["estimator"](),
            grid,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        search.fit(X_train, y)
        best_estimator = search.best_estimator_
        tuned = {
            "best_params": search.best_params_,
            "cv_log_rmse": -search.best_score_,
        }
        print(f"  tuned CV log-RMSE: {tuned['cv_log_rmse']:.4f} | params: {tuned['best_params']}")
    else:
        best_estimator = specs[best_name]["estimator"]()
        best_estimator.fit(X_train, y)

    # --- Reproducible artifacts --------------------------------------------
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = save_dir / "reports"
        figures_dir = reports_dir / "figures"
        processed_dir = save_dir / "data" / "processed"
        models_dir = save_dir / "models"
        for d in (reports_dir, figures_dir, processed_dir, models_dir):
            d.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "model": best_estimator,
                "feature_columns": feature_columns,
                "target_transform": "log1p",
            },
            models_dir / "best_model.joblib",
        )

        # Per-fold scores, long format
        fold_rows = [
            {"model": name, "fold": fold, "log_rmse": score}
            for name, r in results.items()
            for fold, score in enumerate(r["scores"], start=1)
        ]
        pd.DataFrame(fold_rows).to_csv(reports_dir / "cv_results.csv", index=False)

        # Test predictions (real test Ids, expm1 to get back to dollars)
        log_pred = best_estimator.predict(X_test)
        ids = test_ids if test_ids is not None else np.arange(len(X_test))
        submission = pd.DataFrame(
            {"Id": ids, "SalePrice": np.expm1(log_pred)}
        )
        submission.to_csv(processed_dir / "test_predictions.csv", index=False)

        _write_report(
            reports_dir / "model_comparison.md",
            results=results,
            summary=summary,
            best_name=best_name,
            tuned=tuned,
            n_splits=n_splits,
            n_predictions=len(submission),
        )
        _plot_comparison(summary, figures_dir / "model_comparison.png")

    return {
        "summary": summary,
        "best_model": best_name,
        "results": results,
        "tuned": tuned,
    }


def train_models(save: bool = True, n_splits: int = 5) -> dict:
    """End-to-end training: load processed data, train, compare, save.

    Returns the same dict as :func:`train_and_compare`.
    """
    train_df, test_df = load_processed_data()
    X_train, X_test, y, feature_columns = prepare_features(train_df, test_df)

    print(f"Features: {len(feature_columns)} | rows: {X_train.shape[0]} | CV folds: {n_splits}\n")
    return train_and_compare(
        X_train,
        X_test,
        y,
        feature_columns,
        test_ids=test_df[ID_COLUMN].to_numpy() if ID_COLUMN in test_df.columns else None,
        save_dir=PROJECT_ROOT if save else None,
        n_splits=n_splits,
    )


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _write_report(
    path: Path,
    *,
    results: dict,
    summary: pd.DataFrame,
    best_name: str,
    tuned: Optional[dict],
    n_splits: int,
    n_predictions: int,
) -> None:
    lines = [
        "# Model comparison — HousePrice-AI (Phase 3)",
        "",
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "- Metric: **log-RMSE** = RMSE between `log1p(SalePrice)` prediction and truth (Kaggle metric)",
        f"- Cross-validation: {n_splits}-fold shuffled KFold (seed {RANDOM_STATE}), trained on `log1p(SalePrice)`",
        "",
        "## Comparison (default configurations)",
        "",
        "| model | mean log-RMSE | std log-RMSE |",
        "|---|--:|--:|",
    ]
    for name, row in summary.iterrows():
        lines.append(f"| {name} | {row['mean_log_rmse']:.4f} | {row['std_log_rmse']:.4f} |")

    lines += [
        "",
        f"## Best model: **{best_name}**",
        f"- Mean CV log-RMSE: {summary.loc[best_name, 'mean_log_rmse']:.4f} "
        f"(+/- {summary.loc[best_name, 'std_log_rmse']:.4f})",
    ]
    if tuned:
        lines += [
            f"- Tuned CV log-RMSE: {tuned['cv_log_rmse']:.4f}",
            f"- Best params: {tuned['best_params']}",
        ]
    lines += [
        "",
        f"## Per-fold log-RMSE",
        "",
        "| model | fold | log-RMSE |",
        "|---|--:|--:|",
    ]
    for name, r in results.items():
        for fold, score in enumerate(r["scores"], start=1):
            lines.append(f"| {name} | {fold} | {score:.4f} |")

    lines += [
        "",
        "## Artifacts",
        "",
        f"- `models/best_model.joblib` — fitted best model + `feature_columns`",
        f"- `reports/cv_results.csv` — per-fold scores (long format)",
        f"- `reports/figures/model_comparison.png` — comparison chart",
        f"- `data/processed/test_predictions.csv` — {n_predictions} test predictions (`Id`, `SalePrice`)",
        "",
        "> These scores come from cross-validation on the training set only. ",
        "> No Kaggle submission has been made, so none of these are leaderboard scores.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_comparison(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    means = summary["mean_log_rmse"]
    stds = summary["std_log_rmse"]
    ax.bar(means.index, means, yerr=stds, capsize=5, color="steelblue", alpha=0.85)
    for name, mean in means.items():
        ax.text(
            means.index.get_loc(name), mean + 0.002,
            f"{mean:.4f}", ha="center", fontsize=8,
        )
    ax.set_ylabel("log-RMSE (lower is better)")
    ax.set_title("Model comparison — 5-fold CV log-RMSE")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    start = time.perf_counter()
    out = train_models(save=True)
    print(f"\nDone in {time.perf_counter() - start:.0f}s")
    print("Artifacts written under:", PATHS.models_dir, PATHS.reports_dir, PATHS.test_predictions)
