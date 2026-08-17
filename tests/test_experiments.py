"""Tests for src.models.experiments (Phase 7 harness).

The experiment stages are run manually (`python -m src.models.experiments
--stage ...`) against the real dataset, so these tests only guard the module
against import/registry breakage — they never run model training.
"""

import pytest

from src.models import experiments
from src.models.experiments import STAGES, _xgb
from src.models.predict import _defaults_from_data, _FALLBACK_DEFAULTS


def test_module_importable_and_registry_complete():
    assert callable(experiments.stage_baseline)
    assert set(STAGES) == {
        "baseline",
        "tune1",
        "tune2",
        "final_xgb",
        "features",
        "ensemble",
        "report",
    }


def test_xgb_factory_returns_xgboost_regressor():
    model = _xgb(n_estimators=50)
    assert type(model).__name__ == "XGBRegressor"
    assert model.n_estimators == 50
    assert model.random_state == 42


def test_defaults_from_data_returns_complete_defaults():
    # When raw data is present, defaults must cover every raw feature column
    # with no NaNs; otherwise the static fallback must be complete.
    defaults = _defaults_from_data() or _FALLBACK_DEFAULTS
    assert len(defaults) == 79  # all raw feature columns (Id/SalePrice excluded)
    assert defaults["OverallQual"] >= 1
