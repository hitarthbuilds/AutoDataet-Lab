import polars as pl

def dataset_overview(df: pl.DataFrame) -> dict:
    return {
        "Rows": df.height,                  # NOT df.height()
        "Columns": df.width,                # NOT df.width()
        "Column Names": df.columns,
        "Numeric Columns": [
            col for col, dt in zip(df.columns, df.dtypes)
            if isinstance(dt, pl.datatypes.NumericType)
        ],
        "Categorical Columns": [
            col for col, dt in zip(df.columns, df.dtypes)
            if dt == pl.Utf8
        ],
    }
