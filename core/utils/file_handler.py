import os
import uuid
import polars as pl

BASE_UPLOAD_DIR = "data/uploaded"
BASE_PROCESSED_DIR = "data/processed"

os.makedirs(BASE_UPLOAD_DIR, exist_ok=True)
os.makedirs(BASE_PROCESSED_DIR, exist_ok=True)


def save_uploaded_file(uploaded_file) -> str:
    file_id = str(uuid.uuid4())[:8]
    file_name = f"{file_id}.parquet"
    save_path = os.path.join(BASE_UPLOAD_DIR, file_name)

    df = pl.read_csv(uploaded_file)
    df.write_parquet(save_path)

    return save_path


def load_parquet(path: str):
    return pl.read_parquet(path)


def get_preview(df, n: int = 10):
    return df.head(n).to_pandas()


def get_basic_info(df):
    return {
        "rows": df.height,
        "columns": df.width,
        "column_names": df.columns,
        "dtypes": [str(t) for t in df.dtypes],
    }
