# Model comparison — HousePrice-AI (Phase 3)

- Generated: 2026-08-17 06:24 UTC
- Metric: **log-RMSE** = RMSE between `log1p(SalePrice)` prediction and truth (Kaggle metric)
- Cross-validation: 5-fold shuffled KFold (seed 42), trained on `log1p(SalePrice)`

## Comparison (default configurations)

| model | mean log-RMSE | std log-RMSE |
|---|--:|--:|
| XGBoost | 0.1272 | 0.0151 |
| GradientBoosting | 0.1309 | 0.0157 |
| RandomForest | 0.1424 | 0.0159 |
| ElasticNet | 0.1436 | 0.0418 |
| Ridge | 0.1460 | 0.0409 |
| DummyBaseline | 0.3997 | 0.0240 |

## Best model: **XGBoost**
- Mean CV log-RMSE: 0.1272 (+/- 0.0151)
- Tuned CV log-RMSE: 0.1270
- Best params: {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 300}

## Per-fold log-RMSE

| model | fold | log-RMSE |
|---|--:|--:|
| DummyBaseline | 1 | 0.4323 |
| DummyBaseline | 2 | 0.3978 |
| DummyBaseline | 3 | 0.3782 |
| DummyBaseline | 4 | 0.4206 |
| DummyBaseline | 5 | 0.3696 |
| Ridge | 1 | 0.1226 |
| Ridge | 2 | 0.1247 |
| Ridge | 3 | 0.2231 |
| Ridge | 4 | 0.1508 |
| Ridge | 5 | 0.1089 |
| ElasticNet | 1 | 0.1207 |
| ElasticNet | 2 | 0.1192 |
| ElasticNet | 3 | 0.2228 |
| ElasticNet | 4 | 0.1483 |
| ElasticNet | 5 | 0.1073 |
| RandomForest | 1 | 0.1454 |
| RandomForest | 2 | 0.1302 |
| RandomForest | 3 | 0.1715 |
| RandomForest | 4 | 0.1385 |
| RandomForest | 5 | 0.1265 |
| GradientBoosting | 1 | 0.1365 |
| GradientBoosting | 2 | 0.1206 |
| GradientBoosting | 3 | 0.1586 |
| GradientBoosting | 4 | 0.1251 |
| GradientBoosting | 5 | 0.1138 |
| XGBoost | 1 | 0.1298 |
| XGBoost | 2 | 0.1210 |
| XGBoost | 3 | 0.1548 |
| XGBoost | 4 | 0.1199 |
| XGBoost | 5 | 0.1106 |

## Artifacts

- `models/best_model.joblib` — fitted best model + `feature_columns`
- `reports/cv_results.csv` — per-fold scores (long format)
- `reports/figures/model_comparison.png` — comparison chart
- `data/processed/test_predictions.csv` — 1459 test predictions (`Id`, `SalePrice`)

> These scores come from cross-validation on the training set only. 
> No Kaggle submission has been made, so none of these are leaderboard scores.
