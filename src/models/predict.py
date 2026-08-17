"""Prediction helpers used by the Streamlit app.

Phase 3/4 TODO:
  - Load the trained model + fitted preprocessor (joblib).
  - Build a DataFrame from the user's inputs and push it through the exact
    same preprocessing/feature pipeline used in training.
  - Return the predicted ``SalePrice`` (and, if available, an interval).
"""

from __future__ import annotations

from typing import Any, Dict


def predict_price(features: Dict[str, Any]) -> float:
    """Predict a sale price from a dict of house characteristics.

    Parameters
    ----------
    features : dict
        User-entered house characteristics keyed by column name.

    Returns
    -------
    float
        Estimated ``SalePrice`` in dollars.

    Raises
    ------
    NotImplementedError
        Implemented in Phase 3/4 once a trained model artifact exists.
    """
    raise NotImplementedError(
        "predict_price() is implemented in Phase 3/4, after a model has been "
        "trained and saved to models/best_model.joblib."
    )
