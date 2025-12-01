import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle

from core.preprocess.pipeline import fit_preprocessor
from core.utils.sessions import get_df

# optional import (safe fallback)
try:
    from core.utils.sessions import set_processed_df
except Exception:
    def set_processed_df(df):
        return None


# ==================================================================
# MAIN PAGE
# ==================================================================
def app():

    st.markdown(
        """
        <h1 style='margin-bottom:0px;'>🛠️ Preprocessing</h1>
        <p style='opacity:0.7;margin-top:-5px;'>Enterprise-grade automated preprocessing engine</p>
        """,
        unsafe_allow_html=True
    )

    df = get_df()
    if df is None:
        st.error("⚠ No dataset loaded.")
        return

    # Convert polars → pandas if needed
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()

    st.subheader("📘 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    st.divider()

    # ==================================================================
    # SIDEBAR CONFIG
    # ==================================================================
    st.sidebar.header("⚙️ Preprocessing Options")

    config = {
        "rare_threshold": st.sidebar.slider("Merge rare categories (freq < x)", 0.0, 0.05, 0.01),

        "imputer_numeric_strategy": st.sidebar.selectbox(
            "Numeric Imputation Strategy",
            ["mean", "median", "most_frequent"]
        ),

        "imputer_categorical_strategy": st.sidebar.selectbox(
            "Categorical Imputation Strategy",
            ["most_frequent", "constant"]
        ),

        "scaler": st.sidebar.selectbox(
            "Numeric Scaling",
            ["standard", "minmax", "robust"]
        ),

        "missing_indicator": st.sidebar.checkbox(
            "Add Missing Value Indicators", True
        ),

        "leakage_corr_threshold": st.sidebar.slider(
            "Correlation leakage threshold", 0.01, 1.0, 0.95
        ),

        "leakage_mi_threshold": st.sidebar.slider(
            "Mutual information leakage threshold", 0.01, 1.0, 0.6
        ),
    }

    run = st.sidebar.button("🚀 Run Preprocessing", use_container_width=True)

    if not run:
        return

    # ==================================================================
    # RUNNING PIPELINE
    # ==================================================================
    with st.spinner("⏳ Running preprocessing pipeline..."):
        try:
            result = fit_preprocessor(df, target_col=None, config=config)
        except Exception as e:
            st.error("❌ Preprocessing failed!")
            st.exception(e)
            return

    st.success("🎉 Preprocessing Completed Successfully!")

    # ==================================================================
    # SUMMARY CARD
    # ==================================================================
    st.markdown("<h3>📊 Preprocessing Summary</h3>", unsafe_allow_html=True)

    before = result.summary.get("input_shape", ("?", "?"))
    after = result.summary.get("processed_shape", ("?", "?"))

    c1, c2 = st.columns(2)
    c1.metric("Rows", before[0])
    c1.metric("Columns (Before)", before[1])
    c2.metric("Columns (After)", after[1])
    c2.metric("Added Features", after[1] - before[1])

    st.divider()

    # ==================================================================
    # LEAKAGE REPORT
    # ==================================================================
    st.markdown("<h3>🕵️ Leakage Detection</h3>", unsafe_allow_html=True)

    leak = result.summary.get("leak_report", {})

    if isinstance(leak, dict) and (leak.get("corr") or leak.get("mi")):
        st.json(leak)
    else:
        st.info("No leakage detected. Dataset looks clean.")

    st.divider()

    # ==================================================================
    # PROCESSED DATA PREVIEW
    # ==================================================================
    st.markdown("<h3>📦 Processed Data (Sample)</h3>", unsafe_allow_html=True)
    st.dataframe(result.processed_df.head(), use_container_width=True)

    # Save processed df to session
    set_processed_df(result.processed_df)

    st.success("📁 Processed dataset saved to session!")

    st.divider()

    # ==================================================================
    # DOWNLOAD CENTER
    # ==================================================================
    st.markdown("<h3>📥 Download Results</h3>", unsafe_allow_html=True)

    # --- Download processed CSV ---
    csv_bytes = result.processed_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Processed CSV",
        data=csv_bytes,
        file_name="processed_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )

    # --- Download summary JSON ---
    summary_bytes = json.dumps(result.summary, indent=4).encode("utf-8")
    st.download_button(
        label="📑 Download Preprocessing Summary (JSON)",
        data=summary_bytes,
        file_name="preprocessing_summary.json",
        mime="application/json",
        use_container_width=True
    )

    # --- Download pipeline PKL ---
    pipeline_bytes = pickle.dumps(result.pipeline)
    st.download_button(
        label="🧠 Download Preprocessing Pipeline (.pkl)",
        data=pipeline_bytes,
        file_name="preprocessing_pipeline.pkl",
        mime="application/octet-stream",
        use_container_width=True
    )


if __name__ == "__main__":
    app()
