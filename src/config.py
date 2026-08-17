"""Central project configuration.

All global constants (file paths, target column, random seed) live here so
every module reads from one place instead of hard-coding strings.
"""

from dataclasses import dataclass
from pathlib import Path

# Root of the project (parent of the `src/` package).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    """Canonical locations for raw data, processed data, and artifacts."""

    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    figures_dir: Path = PROJECT_ROOT / "reports" / "figures"

    train_raw: Path = raw_data_dir / "train.csv"
    test_raw: Path = raw_data_dir / "test.csv"

    train_processed: Path = processed_data_dir / "train_processed.csv"
    test_processed: Path = processed_data_dir / "test_processed.csv"

    best_model: Path = models_dir / "best_model.joblib"
    preprocessor: Path = models_dir / "preprocessor.joblib"

    # Phase 3 artifacts
    cv_results: Path = PROJECT_ROOT / "reports" / "cv_results.csv"
    model_report: Path = PROJECT_ROOT / "reports" / "model_comparison.md"
    comparison_figure: Path = PROJECT_ROOT / "reports" / "figures" / "model_comparison.png"
    test_predictions: Path = PROJECT_ROOT / "data" / "processed" / "test_predictions.csv"


# Global constants used across the pipeline.
TARGET: str = "SalePrice"  # Kaggle target column
ID_COLUMN: str = "Id"  # Identifier column, excluded from modelling
RANDOM_STATE: int = 42  # Reproducibility seed

# Default paths instance used throughout the project.
PATHS = Paths()
