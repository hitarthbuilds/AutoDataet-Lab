import polars as pl

def correlation_matrix(df: pl.DataFrame) -> pl.DataFrame:
    numeric_df = df.select(pl.col(pl.datatypes.NumericType))
    return numeric_df.corr()
