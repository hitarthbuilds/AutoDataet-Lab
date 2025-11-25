import polars as pl

def dataset_overview(df: pl.DataFrame) -> dict:

    numeric_cols = [col for col, dt in zip(df.columns, df.dtypes) if dt in pl.NUMERIC_DTYPES]
    categorical_cols = [col for col, dt in zip(df.columns, df.dtypes) if dt == pl.Utf8]

    return {
        "Rows": df.height,
        "Columns": df.width,
        "Column Names": df.columns,
        "Dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
        "Numeric Columns": numeric_cols,
        "Categorical Columns": categorical_cols
    }
