# houseprice-ai
End-to-end machine learning solution for the Kaggle House Prices competition, covering data preprocessing, exploratory data analysis, feature engineering, regression model comparison, cross-validation, hyperparameter tuning, and model deployment.

> 🚧 **Status: Phase 3 complete (model training + comparison).** Models are
> trained and compared with cross-validation on the real dataset; the best
> model is saved. Scores below are CV scores on the training set — **no Kaggle
> submission has been made** and no leaderboard ranking is claimed.

## What is this project?

A professional, portfolio-grade ML project built around Kaggle's
**[House Prices — Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)**
competition (Ames Housing dataset). The goal is to predict the sale price of
homes in Ames, Iowa from 79 explanatory variables.

**Evaluation metric (Kaggle):** Root Mean Squared Error (RMSE) between the
logarithm of the predicted price and the logarithm of the actual `SalePrice`:

```text
RMSE(log) = sqrt( mean( (log(1 + y_true) - log(1 + y_pred)) ** 2 ) )
```

## Project goals

1. Explore and preprocess the Ames Housing dataset.
2. Perform meaningful EDA and missing-value analysis.
3. Engineer useful housing features (total square footage, total baths, house age, ...).
4. Train and compare multiple regression models (linear/ridge baseline, Random Forest, Gradient Boosting, XGBoost).
5. Use cross-validation honestly (no train/test leakage).
6. Optimize for RMSE on `log(SalePrice)`.
7. Save the best trained model.
8. Ship a Streamlit app for interactive price estimation.
9. Prepare for containerization with Docker.
10. Keep the code clean, modular, and easy to follow.

## Repository structure

```text
HousePrice-AI/
├── app.py                    # Streamlit app (Phase 1 skeleton, no fake predictions)
├── requirements.txt          # Python 3.10 dependencies
├── pytest.ini                # pytest configuration
├── data/                     # Dataset directory (contents NOT committed to git)
│   ├── raw/                  #   <- train.csv and test.csv (official Kaggle data)
│   ├── processed/            #   <- cleaned/engineered CSVs written by the pipeline
│   └── README.md
├── models/                   # Trained artifacts (joblib) - NOT committed
├── notebooks/
│   └── 01_eda.ipynb          # Exploratory data analysis (run for real)
├── reports/
│   ├── cv_results.csv        # Per-fold log-RMSE of every model (Phase 3)
│   ├── model_comparison.md   # Model comparison report (Phase 3)
│   └── figures/              # EDA + comparison figures
├── src/
│   ├── config.py             # Central paths & constants
│   ├── pipeline.py           # Phase 2 pipeline: load -> preprocess -> engineer -> save
│   ├── data/
│   │   └── load_data.py      # Load raw / processed CSVs with clear errors
│   ├── features/
│   │   ├── preprocessing.py       # Cleaning, missing values, ordinal encodings
│   │   └── feature_engineering.py # Domain features (TotalSF, HouseAge, ...)
│   ├── models/
│   │   ├── train.py          # Cross-validated training + comparison (Phase 3)
│   │   └── predict.py        # Load model & predict (Phase 4)
│   └── evaluation/
│       └── evaluate.py       # RMSE / RMSE-on-log metrics + CV helper
└── tests/                    # pytest suite (unit tests + placeholders)
```

## Getting started

### 1. Clone & environment

```bash
git clone <your-repo-url> HousePrice-AI
cd HousePrice-AI
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add the dataset

Download the official `train.csv` and `test.csv` from the
[Kaggle competition data page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
and place them in `data/raw/`. See `data/README.md` for details.

> 🔒 Datasets, model binaries, and secrets are **gitignored** and will never be
> committed.

### 3. Run the processing pipeline (Phase 2)

```bash
python -m src.pipeline
```

Loads the raw CSVs, applies preprocessing + feature engineering, and writes
NaN-free data to `data/processed/`. Then explore the EDA notebook:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 4. Run the tests

```bash
pytest
```

### 5. Train and compare models (Phase 3)

```bash
python -m src.models.train
```

Cross-validates all candidate models on `log1p(SalePrice)` (5-fold, seeded),
tunes the winner, and writes:

- `models/best_model.joblib` — fitted best model + feature column order
- `reports/cv_results.csv` — per-fold log-RMSE of every model
- `reports/model_comparison.md` — comparison report
- `reports/figures/model_comparison.png` — comparison chart
- `data/processed/test_predictions.csv` — 1459 test predictions (`Id`, `SalePrice`)

### 6. Run the Streamlit app (skeleton)

```bash
streamlit run app.py
```

The app currently reports pipeline status only. The price-estimation form stays
**disabled** until a trained model artifact exists — no made-up numbers.

## Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Project scaffolding (structure, config, module placeholders, tests, Streamlit skeleton) | ✅ |
| 2 | EDA, preprocessing, feature engineering | ✅ (`python -m src.pipeline`) |
| 3 | Model training, cross-validation, tuning, save best model | ✅ (`python -m src.models.train`) |
| 4 | Wire the Streamlit app to real predictions | ⏳ |
| 5 | Docker image + final polish | ⏳ |

## Phase 3 results (cross-validation on the training set)

Metric: **log-RMSE** = RMSE between `log1p(SalePrice)` prediction and truth
(5-fold shuffled KFold, seed 42, trained on `log1p(SalePrice)`). Full detail in
[`reports/model_comparison.md`](reports/model_comparison.md).

| model | mean log-RMSE | std |
|---|--:|--:|
| **XGBoost** (tuned) | **0.1270** | — |
| XGBoost (default) | 0.1272 | 0.0151 |
| GradientBoosting | 0.1309 | 0.0157 |
| RandomForest | 0.1424 | 0.0159 |
| ElasticNet | 0.1436 | 0.0418 |
| Ridge | 0.1460 | 0.0409 |
| DummyBaseline | 0.3997 | 0.0240 |

> These are **cross-validation scores, not Kaggle leaderboard scores**. No
> submission has been made yet. Best model: XGBoost (`learning_rate=0.05`,
> `max_depth=3`, `n_estimators=300`), saved to `models/best_model.joblib`.

## Honesty policy

- No metric, accuracy, or Kaggle score will be reported unless it comes from
  actually running the pipeline on the real dataset.
- No Kaggle ranking will be claimed unless a real submission exists.
- The Streamlit app will never show fabricated estimates.

## License

MIT (add a LICENSE file if you publish the repository).
