import polars as pl
import os

DATA_PATH = "data/current.csv"

def save_uploaded_file(uploaded_file):
    file_path = DATA_PATH

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def load_dataset(path=DATA_PATH):
    try:
        df = pl.read_csv(
            path,
            infer_schema_length=200000,
            ignore_errors=True,
            null_values=["", " ", "NA", "N/A", "-", ".", "...", "null"]
        )
    except Exception:
        # ultra-safe fallback: read everything as string
        df = pl.read_csv(
            path,
            dtypes={"*": pl.Utf8},
            ignore_errors=True
        )

    return df
