import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def missing_summary(df: pl.DataFrame) -> pd.DataFrame:
    """
    Returns missing value statistics for each column.
    """
    total_rows = df.height()
    missing_counts = df.null_count()

    summary = pd.DataFrame({
        "Column": df.columns,
        "Missing": missing_counts.values(),
        "Missing %": [
            round((missing / total_rows) * 100, 2)
            for missing in missing_counts.values()
        ],
        "Dtype": [str(df[col].dtype) for col in df.columns]
    })

    summary.sort_values(by="Missing", ascending=False, inplace=True)
    return summary


def missing_heatmap(pdf: pd.DataFrame):
    """
    Generates a heatmap showing missing value patterns across rows & columns.
    """
    if pdf.isnull().sum().sum() == 0:
        return None  # no missing values at all

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        pdf.isnull(),
        cbar=False,
        yticklabels=False
    )
    plt.title("Missing Value Heatmap")
    plt.tight_layout()
    return plt
