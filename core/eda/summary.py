import polars as pl

def dataset_overview(df: pl.DataFrame) -> dict:
    return {
        "Rows": df.height,                 # Polars property
        "Columns": df.width,               # Polars property
        "Column Names": df.columns,
        "Dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
        "Numeric Columns": [col for col, dt in zip(df.columns, df.dtypes) if dt.is_numeric()],
        "Categorical Columns": [col for col, dt in zip(df.columns, df.dtypes) if dt.is_utf8()],
    }
