import streamlit as st
import pandas as pd
from core.utils.file_handler import save_uploaded_file, load_dataset

st.set_page_config(
    page_title="Upload Dataset",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Upload Dataset")
st.caption("Upload a CSV file up to 200MB.")

uploaded_file = st.file_uploader("Browse CSV file", type=["csv"])

if uploaded_file:
    # save uploaded file to data/current.csv
    saved_path = save_uploaded_file(uploaded_file)
    
    # store in session state
    st.session_state["uploaded_file"] = uploaded_file
    st.session_state["uploaded_path"] = saved_path

    # load dataset with safe Polars loader
    df = load_dataset(saved_path)

    st.success("Dataset uploaded successfully!")
    
    st.subheader("Preview")
    st.dataframe(df.head(50))

    st.info(
        f"Rows: {df.shape[0]} | Columns: {df.shape[1]}"
    )
