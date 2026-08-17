"""HousePrice-AI — Streamlit front end.

Phase 1: UI skeleton only. Price prediction is intentionally NOT implemented
until a trained model artifact exists (Phase 3/4), so users are never shown
fabricated estimates.

Run from the project root:

    streamlit run app.py
"""

import streamlit as st

from src.config import PATHS

st.set_page_config(page_title="HousePrice-AI", page_icon="🏠", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "HousePrice-AI: an end-to-end ML project for the Kaggle "
        "**House Prices — Advanced Regression Techniques** competition "
        "(Ames Housing dataset)."
    )
    st.markdown(
        "[Kaggle competition page](https://www.kaggle.com/competitions/"
        "house-prices-advanced-regression-techniques)"
    )
    st.divider()
    st.caption("Phase 1 scaffolding — no predictions yet.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏠 HousePrice-AI")
st.subheader("Estimate house prices in Ames, Iowa")

# ---------------------------------------------------------------------------
# Pipeline status
# ---------------------------------------------------------------------------
st.subheader("Pipeline status")

dataset_ready = PATHS.train_raw.is_file() and PATHS.test_raw.is_file()
model_ready = PATHS.best_model.is_file()

col1, col2, col3 = st.columns(3)
col1.metric(
    "Kaggle dataset",
    "✅ ready" if dataset_ready else "❌ missing",
    help=f"Expected at {PATHS.raw_data_dir}",
)
col2.metric(
    "Trained model",
    "✅ ready" if model_ready else "❌ not trained",
    help=f"Expected at {PATHS.best_model}",
)
col3.metric("Streamlit UI", "✅ skeleton", help="Prediction wiring lands in Phase 4")

if not dataset_ready:
    st.info(
        "The official Kaggle dataset has not been added yet. "
        "Download `train.csv` and `test.csv` from the competition page and "
        f"place them in `{PATHS.raw_data_dir}` (see `data/README.md`)."
    )

st.divider()

# ---------------------------------------------------------------------------
# Price estimation form (placeholder - disabled until a model exists)
# ---------------------------------------------------------------------------
st.subheader("Estimate a price")

if not model_ready:
    st.warning(
        "No trained model found. This form is **disabled on purpose** — the "
        "project will not show made-up estimates. It becomes live in Phase 3/4 "
        "once a model has been trained and saved."
    )
    enabled = False
else:
    enabled = True

with st.form("house_form"):
    st.caption(
        "Sample house characteristics (full set of 79 features comes in Phase 4)."
    )
    overall_qual = st.slider(
        "Overall quality (1–10)", min_value=1, max_value=10, value=5, disabled=not enabled
    )
    total_sqft = st.number_input(
        "Total square footage", min_value=0, value=1500, step=50, disabled=not enabled
    )
    year_built = st.number_input(
        "Year built", min_value=1800, max_value=2025, value=1970, disabled=not enabled
    )
    submitted = st.form_submit_button(
        "Estimate price", disabled=not enabled, type="primary" if enabled else "secondary"
    )

    if submitted:
        # Phase 4 TODO: call src.models.predict.predict_price(...) with the
        # collected inputs. Not wired yet - never return fake numbers.
        st.error("Prediction wiring is implemented in Phase 4.")
