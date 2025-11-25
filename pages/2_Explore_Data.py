import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from core.utils.file_handler import load_dataset

st.set_page_config(
    page_title="Explore Data",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Explore Data")

# ----------------------------------
# Load dataset from previous upload
# ----------------------------------
uploaded = st.session_state.get("uploaded_file", None)
if not uploaded:
    st.warning("Please upload a dataset first from the 'Upload Dataset' page.")
    st.stop()

# load polars df then convert to pandas
pl_df = load_dataset()
df = pl_df.to_pandas()   # use pandas for all analysis

# ----------------------------------
# Tabs
# ----------------------------------
tab_overview, tab_missing, tab_dist, tab_corr = st.tabs(
    ["Overview", "Missing Values", "Distributions", "Correlations"]
)

# ----------------------------------
# OVERVIEW TAB
# ----------------------------------
with tab_overview:
    st.subheader("Dataset Overview")

    overview = {
        "Rows": int(df.shape[0]),
        "Columns": int(df.shape[1]),
        "Column Names": list(df.columns),
        "Dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
        "Numeric Columns": [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
        ],
        "Categorical Columns": [
            col for col in df.columns
            if pd.api.types.is_string_dtype(df[col])
        ],
    }

    st.json(overview)

    st.subheader("Preview (First 100 Rows)")
    st.dataframe(df.head(100), use_container_width=True)


# ----------------------------------
# MISSING VALUES TAB
# ----------------------------------
with tab_missing:
    st.subheader("Missing Value Summary")

    total_rows = df.shape[0]
    missing_counts = df.isna().sum()
    missing_percent = (missing_counts / total_rows * 100).round(2)

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing": missing_counts.values,
        "Missing %": missing_percent.values,
        "Dtype": [str(dt) for dt in df.dtypes]
    }).sort_values("Missing", ascending=False)

    st.dataframe(missing_df, use_container_width=True)

    st.subheader("Missing Value Heatmap")

    if missing_counts.sum() == 0:
        st.success("No missing values found in this dataset.")
    else:
        # show only first 1000 rows for readability
        sample = df.head(1000)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(sample.isna(), aspect="auto", interpolation="nearest")
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows (first 1000)")
        ax.set_title("Missing Value Pattern (yellow = missing)")
        ax.set_xticks(range(len(sample.columns)))
        ax.set_xticklabels(sample.columns, rotation=90, fontsize=8)
        ax.set_yticks([])
        st.pyplot(fig)


# ----------------------------------
# DISTRIBUTIONS TAB
# ----------------------------------
with tab_dist:
    st.subheader("Column Distribution")

    column = st.selectbox("Select a column", df.columns)

    if column:
        series = df[column].dropna()

        if series.empty:
            st.warning("Column has only missing values.")
        else:
            fig, ax = plt.subplots(figsize=(8, 4))

            if pd.api.types.is_numeric_dtype(series):
                ax.hist(series, bins=40)
                ax.set_title(f"Distribution of {column}")
                ax.set_xlabel(column)
                ax.set_ylabel("Frequency")
            else:
                # categorical / text: show top 30 categories
                value_counts = series.value_counts().head(30)
                ax.bar(value_counts.index.astype(str), value_counts.values)
                ax.set_title(f"Top values in {column}")
                ax.set_xticklabels(value_counts.index.astype(str), rotation=90, fontsize=8)
                ax.set_ylabel("Count")

            plt.tight_layout()
            st.pyplot(fig)


# ----------------------------------
# CORRELATIONS TAB
# ----------------------------------
with tab_corr:
    st.subheader("Correlation Heatmap (Numeric Columns Only)")

    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        st.warning("Not enough numeric columns to compute correlations.")
    else:
        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        cax = ax.imshow(corr.values, interpolation="nearest", cmap="coolwarm")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        fig.colorbar(cax)
        ax.set_title("Correlation Heatmap")
        plt.tight_layout()
        st.pyplot(fig)
