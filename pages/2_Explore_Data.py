import streamlit as st
import pandas as pd
import polars as pl

from core.utils.file_handler import load_dataset
from core.eda.summary import dataset_overview
from core.eda.missing_value import missing_summary, missing_heatmap
from core.eda.correlations import compute_correlation
from core.eda.visualisation import plot_distribution, plot_heatmap


st.set_page_config(
    page_title="Explore Data",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Explore Data")

# Check if dataset is uploaded
uploaded = st.session_state.get("uploaded_file", None)
if not uploaded:
    st.warning("Please upload a dataset first from the 'Upload Dataset' page.")
    st.stop()

# Load dataset (Polars)
df = load_dataset()

# Convert to Pandas for plotting
pdf = df.to_pandas()


# ------------------------------
#  TABS
# ------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Missing Values",
    "Distributions",
    "Correlations"
])


# ------------------------------
#  TAB 1 — OVERVIEW
# ------------------------------
with tab1:
    st.subheader("Dataset Overview")
    overview = dataset_overview(df)
    st.json(overview)

    st.subheader("Preview (First 100 Rows)")
    st.dataframe(pdf.head(100), use_container_width=True)


# ------------------------------
#  TAB 2 — MISSING VALUES
# ------------------------------
with tab2:
    st.subheader("Missing Value Summary")
    summary_df = missing_summary(df)
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("Missing Value Heatmap")
    fig = missing_heatmap(pdf)

    if fig:
        st.pyplot(fig)
    else:
        st.success("No missing values found!")


# ------------------------------
#  TAB 3 — DISTRIBUTIONS
# ------------------------------
with tab3:
    st.subheader("Column Distribution")

    col = st.selectbox("Select a column", df.columns)

    fig = plot_distribution(pdf, col)
    if fig:
        st.pyplot(fig)
    else:
        st.warning("This column cannot be visualized (likely non-numeric or too complex).")


# ------------------------------
#  TAB 4 — CORRELATIONS
# ------------------------------
with tab4:
    st.subheader("Correlation Heatmap (Numeric Columns Only)")

    corr = compute_correlation(pdf)

    if corr is None:
        st.warning("No numeric columns available for correlation.")
    else:
        fig = plot_heatmap(corr)
        st.pyplot(fig)
