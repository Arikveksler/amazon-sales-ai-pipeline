"""
Amazon Sales AI Pipeline - Streamlit Dashboard
Week 3: UI Developer Implementation
"""

import time

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path

# Project root (src/ -> parent)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Output file paths
INSIGHTS_PATH = PROJECT_ROOT / "insights.md"
EDA_REPORT_PATH = PROJECT_ROOT / "data" / "eda_report.html"

# Columns the pipeline (EDA + feature engineering) requires in the uploaded CSV
REQUIRED_COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating",
    "rating_count",
]


def load_text_file(path: Path) -> str | None:
    """Read a text file if it exists, return None otherwise."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Amazon Sales AI Pipeline",
    page_icon="📊",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("Amazon Sales AI Pipeline")
st.sidebar.markdown("CrewAI-powered analysis of Amazon India product data.")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload Amazon Sales CSV", type=["csv"])

st.sidebar.markdown("---")

# TODO ARIK: Trigger actual CrewAI flow here
run_pipeline = st.sidebar.button("Run Analysis Pipeline")
if run_pipeline:
    with st.spinner("Running Analyst Crew..."):
        time.sleep(2)
    st.sidebar.success("Pipeline execution complete!")

st.sidebar.markdown("---")
st.sidebar.caption("Output files")
st.sidebar.text(f"Insights:  {'Found' if INSIGHTS_PATH.exists() else 'Not found'}")
st.sidebar.text(f"EDA Report: {'Found' if EDA_REPORT_PATH.exists() else 'Not found'}")

# ── Main Title ───────────────────────────────────────────────────────────────

st.title("Amazon Sales AI Pipeline")

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "Data Preview",
    "Insights",
    "Visualizations",
    "Prediction",
])

# ── Tab 1: Data Preview ─────────────────────────────────────────────────────

with tab1:
    st.header("Data Preview")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            st.error(
                "**Invalid CSV — missing required columns:**\n\n"
                + "\n".join(f"- `{c}`" for c in missing_cols)
                + "\n\nPlease upload the Amazon Sales CSV with the correct schema."
            )
            st.stop()

        st.success(f"Loaded **{len(df)}** rows and **{len(df.columns)}** columns.")
        st.dataframe(df.head(), use_container_width=True)

        with st.expander("Column Info"):
            col_info = pd.DataFrame({
                "Type": df.dtypes.astype(str),
                "Non-Null": df.notna().sum(),
                "Null": df.isna().sum(),
            })
            st.dataframe(col_info, use_container_width=True)
    else:
        st.info("Upload a CSV file using the sidebar to preview the data.")

# ── Tab 2: Insights ─────────────────────────────────────────────────────────

with tab2:
    st.header("Business Insights")

    # Check root first, then data/ as a fallback
    _insights_candidates = [
        INSIGHTS_PATH,
        PROJECT_ROOT / "data" / "insights.md",
    ]
    content = next(
        (load_text_file(p) for p in _insights_candidates if p.exists()),
        None,
    )
    if content:
        st.markdown(content)
    else:
        st.info("No insights generated yet. Run the pipeline first.")

# ── Tab 3: Visualizations ───────────────────────────────────────────────────

with tab3:
    st.header("EDA Report")

    content = load_text_file(EDA_REPORT_PATH)
    if content:
        components.html(content, height=800, scrolling=True)
    else:
        st.warning(
            f"File not found: `{EDA_REPORT_PATH.relative_to(PROJECT_ROOT)}`\n\n"
            "Run the Analyst Crew first:\n```bash\npython main.py\n```"
        )

# ── Tab 4: Prediction ───────────────────────────────────────────────────────

with tab4:
    st.header("Prediction")
    st.markdown("Enter product details below to get a predicted rating.")

    with st.form("prediction_form"):
        price = st.number_input("Discounted Price (₹)", min_value=0.0, value=499.0, step=50.0)
        category = st.selectbox("Category", [
            "Electronics", "Computers&Accessories", "Home&Kitchen",
            "Health&PersonalCare", "Toys&Games", "Other",
        ])
        review_count = st.number_input("Number of Reviews", min_value=0, value=500, step=100)
        discount_pct = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=40.0, step=5.0)

        submitted = st.form_submit_button("Predict Rating")

    if submitted:
        # ================================================================
        # TODO NAVEH: Load model.pkl here and replace this mock result
        # with model.predict(input_data).
        #
        # Example integration:
        #   import joblib, numpy as np
        #   model = joblib.load("outputs/models/model.pkl")
        #   input_data = np.array([[price, discount_pct, review_count, ...]])
        #   prediction = model.predict(input_data)[0]
        # ================================================================
        st.success("Mock Prediction: Expected Rating is 4.6 ⭐")
        st.info("This is a placeholder result. Connect model.pkl to get real predictions.")
