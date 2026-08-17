# Kaggle submission report - HousePrice-AI (Phase 4)

- Generated: 2026-08-17 06:30 UTC
- Model: `models/best_model.joblib` (XGBoost, trained on `log1p(SalePrice)`)
- Submission file: `C:\Users\SaiRam\Desktop\houseprice-ai\data\processed\kaggle_submission.csv`

## Validation summary

| check | value |
|---|---|
| Rows | 1459 |
| Columns | Id, SalePrice |
| Minimum predicted price | $40,906 |
| Maximum predicted price | $529,229 |
| Mean predicted price | $177,499 |
| Missing predictions | 0 |
| NaN / infinite / negative values | none |

## First 5 rows

| Id | SalePrice |
|---|---|
| 1461 | 120004.34 |
| 1462 | 160315.50 |
| 1463 | 175800.91 |
| 1464 | 191369.44 |
| 1465 | 184743.67 |

> Predictions come from the trained model only - no manual adjustment.
> Format: header `Id,SalePrice` with 1459 data rows (Kaggle-compatible).
