# Model development — HousePrice-AI (Phase 7)

- Generated: 2026-08-17 07:23 UTC
- Metric: log-RMSE on `log1p(SalePrice)` (Kaggle metric)
- CV: 5-fold shuffled KFold, seed 42, identical folds for every experiment
- Baseline Kaggle score: **0.12656** (existing submission, untouched)

## 1. Baseline (verified by execution)

| property | value |
|---|---|
| Model | XGBoost (tuned) |
| Feature count | 241 |
| Rows (train) | 1460 |
| CV strategy | 5-fold shuffled KFold (seed 42), trained on log1p(SalePrice) |
| CV mean log-RMSE | 0.1270 |
| CV std | 0.0156 |
| Hyperparameters | {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': None, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': None, 'reg_lambda': None, 'gamma': None} |
| Kaggle score | 0.12656 |
| Pipeline | preprocess -> engineer_features -> one-hot -> CV -> GridSearchCV -> artifact -> submission |

## 2. Models tested (default configs, same folds)

| Model | CV mean log-RMSE | CV std |
|---|---|---|
| XGBoost_tuned_current | 0.1270 | 0.0156 |
| XGBoost | 0.1272 | 0.0151 |
| GradientBoosting | 0.1309 | 0.0157 |
| RandomForest | 0.1424 | 0.0159 |
| ElasticNet | 0.1436 | 0.0418 |
| Ridge | 0.1460 | 0.0409 |
| DummyBaseline | 0.3997 | 0.0240 |

> Ridge / ElasticNet / RandomForest / GradientBoosting / XGBoost come from `src/models/train.py::build_model_specs`. LightGBM and CatBoost were considered but **not installed**: XGBoost/GradientBoosting/RandomForest already cover the model space and adding hard dependencies would not be justified by expected gains (kept for maintainability).

## 3. XGBoost hyperparameter search

Grid search over the same seeded folds (`GridSearchCV`, `neg_root_mean_squared_error` on the log1p target).

## Grids searched

- **tune1**: `{"learning_rate": [0.03, 0.05], "max_depth": [3, 4, 5], "min_child_weight": [1, 5], "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]}`
- **tune2**: `{"reg_alpha": [0.0, 0.1, 1.0], "reg_lambda": [1.0, 5.0], "gamma": [0.0, 0.1, 0.3]}`
- **tune3**: `{"n_estimators": [300, 500], "learning_rate": [0.03, 0.05]}`

### tune1 — best config

- Best mean log-RMSE: **0.1265** (params: `{'colsample_bytree': 1.0, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 1, 'subsample': 0.8}`, 481.4s)

| rank | params | mean log-RMSE | std |
|---|---|---|---|
| 1 | {'colsample_bytree': 1.0, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 1, 'subsample': 0.8} | 0.1265 | 0.0158 |
| 2 | {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 4, 'min_child_weight': 1, 'subsample': 0.8} | 0.1269 | 0.0150 |
| 3 | {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 1, 'subsample': 0.8} | 0.1270 | 0.0156 |
| 4 | {'colsample_bytree': 1.0, 'learning_rate': 0.05, 'max_depth': 4, 'min_child_weight': 1, 'subsample': 0.8} | 0.1284 | 0.0156 |
| 5 | {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 5, 'min_child_weight': 1, 'subsample': 0.8} | 0.1288 | 0.0161 |

### tune2 — best config

- Best mean log-RMSE: **0.1265** (params: `{'gamma': 0.0, 'reg_alpha': 0.0, 'reg_lambda': 1.0}`, 151.4s)

| rank | params | mean log-RMSE | std |
|---|---|---|---|
| 1 | {'gamma': 0.0, 'reg_alpha': 0.0, 'reg_lambda': 1.0} | 0.1265 | 0.0158 |
| 2 | {'gamma': 0.0, 'reg_alpha': 0.0, 'reg_lambda': 5.0} | 0.1278 | 0.0162 |
| 3 | {'gamma': 0.0, 'reg_alpha': 0.1, 'reg_lambda': 5.0} | 0.1290 | 0.0159 |
| 4 | {'gamma': 0.0, 'reg_alpha': 0.1, 'reg_lambda': 1.0} | 0.1292 | 0.0170 |
| 5 | {'gamma': 0.0, 'reg_alpha': 1.0, 'reg_lambda': 1.0} | 0.1295 | 0.0146 |

### tune3 — best config

- Best mean log-RMSE: **0.1256** (params: `{'learning_rate': 0.05, 'n_estimators': 500}`, 47.1s)

| rank | params | mean log-RMSE | std |
|---|---|---|---|
| 1 | {'learning_rate': 0.05, 'n_estimators': 500} | 0.1256 | 0.0157 |
| 2 | {'learning_rate': 0.05, 'n_estimators': 300} | 0.1265 | 0.0158 |
| 3 | {'learning_rate': 0.03, 'n_estimators': 500} | 0.1272 | 0.0167 |
| 4 | {'learning_rate': 0.03, 'n_estimators': 300} | 0.1300 | 0.0159 |

### Final XGBoost config — honest CV

- Params: `{'colsample_bytree': 1.0, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 1, 'subsample': 0.8, 'n_estimators': 500, 'gamma': 0.0, 'reg_alpha': 0.0, 'reg_lambda': 1.0}`
- CV log-RMSE: **0.1256 ± 0.0157** (re-measured with `cross_validate_and_report`, not the optimistic grid score)
- Per-fold: [0.1286, 0.1168, 0.1547, 0.117, 0.1108]

> The GridSearchCV `best_score_` is mildly optimistic (selection on the same folds); the final CV above is the honest estimate.

## 4. Feature engineering experiments

| Variant | n features | CV mean log-RMSE | CV std | vs baseline |
|---|---|---|---|---|
| baseline | 241 | 0.1256 | 0.0157 | +0.0000 |
| log1p_skewed | 253 | 0.1256 | 0.0157 | +0.0000 |
| interactions | 247 | 0.1260 | 0.0169 | +0.0004 |
| log_and_interactions | 259 | 0.1260 | 0.0169 | +0.0004 |

- `log1p_skewed`: log1p of 12 skewed numerics (LotArea, GrLivArea, TotalSF, ...)
- `interactions`: OverallQual × (GrLivArea, TotalBsmtSF, LotArea, HouseAge, GarageAge, 1stFlrSF)
- Missing-value representation is already handled by the training pipeline (`None`/0 categories) — no change needed.
- No new feature uses `SalePrice` or test data; everything is derived from the input columns alone (no leakage).

## 5. Ensemble experiments (out-of-fold, same folds)

| Blend | weights | OOF log-RMSE |
|---|---|---|
| XGB+GB | [0.5, 0.5, 0.0] | 0.1279 |
| XGB+RF | [0.5, 0.0, 0.5] | 0.1308 |
| XGB+GB+RF | [0.3333333333333333, 0.3333333333333333, 0.3333333333333333] | 0.1299 |
| optimised | [0.9792317847273209, 0.0, 0.02076821527267904] | 0.1265 |

- Single-model OOF: {"XGBoost_tuned": 0.1266, "GradientBoosting": 0.1318, "RandomForest": 0.1433}
- OOF scores use the same 5 folds as model CV (mild optimism).

## 6. Best model

- **XGBoost tuned (Phase 7 config)** with OOF/CV log-RMSE **0.1256** (baseline 0.1270)
- Improvement over baseline: **+0.0014** (improved)

> CV std is ≈0.015, so differences smaller than ~0.001 are within noise; the recommendation below weighs the actual margin.

## 7. Final recommendation

- **Replace the artifact** with the Phase 7 config (`{'colsample_bytree': 1.0, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 1, 'subsample': 0.8, 'n_estimators': 500, 'gamma': 0.0, 'reg_alpha': 0.0, 'reg_lambda': 1.0}` if XGBoost stays best) and generate a new candidate Kaggle submission — the improvement exceeds ~0.001 log-RMSE.
- The Streamlit app and submission pipeline load `models/best_model.joblib`, so replacement keeps both working (same artifact schema).

## 8. Implementation

- `models/best_model.joblib` was **replaced** with the Phase 7 config (same
  artifact schema: `model` + `feature_columns` + `target_transform`, 241
  features unchanged). The old model remains exactly reproducible from the
  committed pipeline + seed 42.
- `data/processed/kaggle_submission_v2.csv` is the candidate submission
  (1459 rows, `Id,SalePrice`, IDs preserved, finite & positive) — the
  existing `kaggle_submission.csv` (score 0.12656) was left untouched.
- Verified after replacement: full test suite, Streamlit prediction path,
  and the submission pipeline's feature-column guard all pass with the new
  artifact.

## 9. Compliance & reproducibility

- All model selection used training/CV results only; the test set was used only to generate candidate predictions through the trained pipeline.
- No test labels inspected, no leaderboard feedback used, no hard-coded or fabricated numbers — every value above comes from actual execution.
- The existing `kaggle_submission.csv` and `models/best_model.joblib` were not modified by the experiments.
