import polars as pl

def missing_summary(df: pl.DataFrame) -> pl.DataFrame:
    total_rows = df.height

    # null_count() returns a DataFrame. Convert to long format.
    null_df = df.null_count().transpose(include_header=True, header_name="Column", value_name="Missing")

    null_df = null_df.with_columns([
        (pl.col("Missing") / total_rows * 100).alias("Percent Missing")
    ])

    return null_df
