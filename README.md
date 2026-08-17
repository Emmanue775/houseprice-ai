# HousePrice-AI

End-to-end machine learning solution for the Kaggle **House Prices —
Advanced Regression Techniques** competition (Ames Housing dataset):
data preprocessing, EDA, feature engineering, model comparison with
cross-validation, hyperparameter tuning, a Kaggle submission, and a
production-ready Streamlit prediction app.

> 🚧 **Status: Phase 6 complete.** The full pipeline is implemented and
> verified: models are cross-validated and tuned on the real dataset, a
> submission was generated from the saved model, and the Streamlit app
> produces live predictions from the trained artifact. A **Kaggle submission
> was made with a private score of 0.12656** (log-RMSE). Docker + CI/CD is
> the remaining roadmap item.

## Problem statement

Given 79 explanatory variables describing residential homes in Ames, Iowa
(quality ratings, square footage, rooms, garage, lot, neighborhood, sale
conditions, ...), predict each home's `SalePrice`.

**Evaluation metric (Kaggle):** Root Mean Squared Error (RMSE) between the
logarithm of the predicted price and the logarithm of the actual `SalePrice`:

```text
RMSE(log) = sqrt( mean( (log(1 + y_true) - log(1 + y_pred)) ** 2 ) )
```

This project trains directly on `log1p(SalePrice)`, so plain RMSE on the
training target is exactly the Kaggle metric.

## Dataset

- **Source:** [Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
- **Train:** 1,460 rows × 81 columns (79 features + `Id` + `SalePrice`)
- **Test:** 1,459 rows × 80 columns
- **Data types:** 43 categorical / 38 numeric columns; 19 columns contain
  missing values — most of which are *not* true missingness (see below).
- Files are **gitignored** (`data/raw/*.csv`, `data/processed/*.csv`) — the
  dataset is never committed.

## EDA

See `notebooks/01_eda.ipynb` and `reports/figures/` (target distribution,
missing values, correlation heatmap, top scatter plots). Key findings:

- `SalePrice` is right-skewed; `log1p` normalizes it.
- `OverallQual`, `GrLivArea`, and `TotalBsmtSF` are the strongest individual
  predictors.
- Most `NA`s mean *"this feature does not exist"* (no alley, no basement, no
  garage, no pool) rather than missing data.

## Preprocessing

Implemented in `src/features/preprocessing.py` (`preprocess`), grounded in the
official data dictionary:

- **"No feature" categories** (e.g. `Alley`, `BsmtQual`, `GarageType`) → the
  category `"None"` instead of imputation (verified: rows with
  `PoolQC`/`BsmtQual`/`GarageType` == NaN always have 0 area).
- **"No feature" numerics** (e.g. `MasVnrArea`, `GarageCars`, `TotalBsmtSF`)
  → `0`.
- **Ordinal ratings** (`Po < Fa < TA < Gd < Ex`) → integers 1–5
  (0 = absent).
- **`LotFrontage`** (the only genuinely missing column) → median of the same
  `Neighborhood`.
- Remaining sparse categoricals → most frequent value (mode).
- `MSSubClass`/`MoSold` are nominal integers → cast to strings.
- A final assertion guarantees **no NaNs survive the pipeline**.

## Feature engineering

Implemented in `src/features/feature_engineering.py` (`engineer_features`),
chosen after EDA:

- **Size:** `TotalSF`, `TotalBath`, `TotalPorchSF`, `TotalOutdoorSF`
- **Age:** `HouseAge`, `RemodAge`, `GarageAge` (0 when no garage)
- **Flags:** `HasGarage`, `HasPool`, `HasFireplace`, `HasBasement`,
  `Has2ndFloor`
- **Interaction:** `QualSF` = `OverallQual × TotalSF`

## Models evaluated

All trained and cross-validated on the **same** seeded 5-fold splits with the
**same** log-RMSE metric (`src/models/train.py`):

| model | mean CV log-RMSE | std |
|---|--:|--:|
| **XGBoost** (tuned) | **0.1270** | — |
| XGBoost (default) | 0.1272 | 0.0151 |
| GradientBoosting | 0.1309 | 0.0157 |
| RandomForest | 0.1424 | 0.0159 |
| ElasticNet | 0.1436 | 0.0418 |
| Ridge | 0.1460 | 0.0409 |
| DummyBaseline | 0.3997 | 0.0240 |

## Cross-validation

5-fold shuffled `KFold` (seed 42) via `src/evaluation/evaluate.py`
(`cross_validate_and_report`). Every fold gets a **fresh, unfitted
estimator** (factory pattern in `build_model_specs`), so there is no leakage.
All models share identical folds and scoring, making the comparison fair.

## Hyperparameter tuning

`GridSearchCV` over the CV winner (`TUNE_GRIDS` in `src/models/train.py`),
scoring `neg_root_mean_squared_error` on the log1p target:

- XGBoost grid: `n_estimators` {150, 300}, `learning_rate` {0.03, 0.05},
  `max_depth` {3, 5}
- **Selected:** `{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 300}`
  → tuned CV log-RMSE **0.1270** (from actual validation, not hard-coded).

The tuned model is refit on all training data and saved to
`models/best_model.joblib` (model + `feature_columns` + `target_transform`).

## Kaggle submission

`src/models/submission.py` loads the saved artifact, re-checks the feature
columns against the training pipeline (guards against drift), predicts all
1,459 test rows, validates the output (`Id`, `SalePrice`, no missing/negative
values), and writes:

- `data/processed/kaggle_submission.csv`
- `reports/submission_report.md`

**Result: private Kaggle score 0.12656** (log-RMSE). Predictions come from the
trained model only — no manual adjustment, no fabricated numbers.

## Streamlit application

`app.py` is a real prediction app (Phase 6):

- Loads the **saved model artifact** (`models/best_model.joblib`) — no
  training at startup.
- Users enter house characteristics through widgets (neighborhood, quality,
  sizes, rooms, garage, ...). Categorical options are derived from the
  artifact's trained column vocabulary — nothing hard-coded.
- Inputs are overlaid on a complete default row (typical values from the
  training set) and pushed through the **exact same** `preprocess` →
  `engineer_features` → one-hot alignment used during training
  (`src/models/predict.py`).
- The model predicts on `log1p(SalePrice)`; the app reverses it with
  `np.expm1` and displays a formatted dollar estimate.
- Invalid inputs are validated and surfaced as errors, never crashes.
- Explicit disclaimers: estimates, not appraisals or guaranteed prices.

```text
User Input → validation → existing feature preparation → saved XGBoost
    → log-space prediction → np.expm1() → predicted SalePrice
```

## How to run locally

```bash
git clone <your-repo-url> HousePrice-AI
cd HousePrice-AI
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Add the dataset: download `train.csv` and `test.csv` from the
[Kaggle competition data page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
into `data/raw/` (see `data/README.md`). Then:

```bash
# 1. Preprocess + feature engineer
python -m src.pipeline

# 2. Train, cross-validate, tune, save the best model
python -m src.models.train

# 3. Generate the Kaggle submission
python -m src.models.submission

# 4. Run the prediction app
streamlit run app.py
```

## Project structure

```text
HousePrice-AI/
├── app.py                    # Streamlit prediction app (Phase 6)
├── requirements.txt
├── pytest.ini
├── data/
│   ├── raw/                  #   train.csv / test.csv (gitignored)
│   └── processed/            #   cleaned + engineered CSVs (gitignored)
├── models/
│   └── best_model.joblib     # trained XGBoost artifact (gitignored)
├── notebooks/01_eda.ipynb
├── reports/                  # CV results, comparison report, submission report, figures
├── src/
│   ├── config.py             # central paths & constants
│   ├── pipeline.py           # raw -> preprocess -> engineer -> save
│   ├── data/load_data.py
│   ├── features/
│   │   ├── preprocessing.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train.py          # CV + tuning + save best model
│   │   ├── predict.py        # inference used by the Streamlit app
│   │   └── submission.py     # Kaggle submission generation
│   └── evaluation/evaluate.py  # RMSE / log-RMSE + CV helper
└── tests/                    # pytest suite
```

## Testing

```bash
pytest
```

36 tests covering preprocessing, feature engineering, evaluation metrics,
training artifacts, the submission pipeline, and the Phase 6 prediction path
(including an end-to-end prediction against the saved artifact). Tests use
tiny synthetic data; artifact-dependent tests skip when the model is absent.

## Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Project structure + EDA | ✅ |
| 2 | Data preprocessing + feature engineering | ✅ |
| 3 | Model training + comparison | ✅ |
| 4 | Cross-validation + hyperparameter tuning | ✅ |
| 5 | Kaggle submission (score 0.12656) | ✅ |
| 6 | Streamlit prediction app | ✅ |
| 7 | Docker + CI/CD | ⏳ |

## Honesty policy

- No metric, accuracy, or Kaggle score is reported unless it comes from
  actually running the pipeline on the real dataset.
- The Kaggle score (0.12656) is from a real submission generated by
  `src/models.submission`.
- The Streamlit app never shows fabricated estimates — without the model
  artifact it refuses to predict, and every prediction comes from the saved
  model via the training-time pipeline.

## License

MIT (add a LICENSE file if you publish the repository).
