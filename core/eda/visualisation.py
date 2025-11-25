import polars as pl

def numeric_distributions(df: pl.DataFrame):
    numeric_cols = [
        col for col, dt in zip(df.columns, df.dtypes)
        if isinstance(dt, pl.datatypes.NumericType)
    ]
    return numeric_cols
