# Data directory

| Path         | Purpose                                                                 |
|--------------|-------------------------------------------------------------------------|
| `raw/`       | Original files from the [Kaggle House Prices competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data): `train.csv` and `test.csv`. |
| `processed/` | Cleaned and feature-engineered DataFrames written by the pipeline.       |

## How to add the dataset

1. Download `train.csv` and `test.csv` from the Kaggle competition page
   (requires a free Kaggle account).
2. Place them directly in `data/raw/`.

## Git policy

These files are **gitignored** and must never be committed: they are large,
and redistributing the competition data violates Kaggle's rules. The
`.gitkeep` files simply keep the folders in version control.
