import streamlit as st
from core.utils.file_handler import (
    save_uploaded_file,
    load_parquet,
    get_preview,
    get_basic_info
)

st.title("Upload Dataset")

if "uploaded_path" not in st.session_state:
    st.session_state.uploaded_path = None

uploaded = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded:
    st.session_state.uploaded_path = save_uploaded_file(uploaded)
    st.success("File uploaded and converted to Parquet successfully.")

if st.session_state.uploaded_path:
    df = load_parquet(st.session_state.uploaded_path)

    st.subheader("Preview")
    st.dataframe(get_preview(df))

    st.subheader("Dataset Info")
    st.json(get_basic_info(df))
