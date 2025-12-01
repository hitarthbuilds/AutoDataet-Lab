from typing import Dict, Optional, Any, NamedTuple
import numpy as np

# Support both pandas + polars seamlessly
try:
    import polars as pl
except:
    pl = None

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)
from sklearn.impute import SimpleImputer

from .config import DEFAULT_CONFIG
from .transformers import ColumnSelector
from .rare_category import RareCategoryMerger
from .missing_pattern import MissingIndicatorAdder
from .leakage import detect_leakage


# -----------------------------
# Helpers
# -----------------------------

def ensure_pandas(df):
    """Convert Polars → Pandas automatically."""
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.to_pandas()
    return df


def scaler_from_name(name: str):
    if name == "minmax":
        return MinMaxScaler()
    if name == "robust":
        return RobustScaler()
    return StandardScaler()


class PreprocessResult(NamedTuple):
    pipeline: Any
    processed_df: pd.DataFrame
    summary: Dict


# -----------------------------
# Build Preprocessor
# -----------------------------

def build_preprocessor(df, target_col=None, config: Optional[Dict] = None):

    df = ensure_pandas(df)
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    df_local = df.copy()

    numerics = df_local.select_dtypes(include=["number"]).columns.tolist()
    categoricals = df_local.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # remove target column
    if target_col in numerics:
        numerics.remove(target_col)
    if target_col in categoricals:
        categoricals.remove(target_col)

    # numeric
    num_pipeline = Pipeline([
        ("select", ColumnSelector(numerics)),
        ("impute", SimpleImputer(strategy=cfg["imputer_numeric_strategy"])),
        ("scale", scaler_from_name(cfg["scaler"]))
    ])

    # categorical
    cat_pipeline = Pipeline([
        ("select", ColumnSelector(categoricals)),
        ("rare", RareCategoryMerger(threshold=cfg["rare_threshold"])),
        ("impute", SimpleImputer(strategy=cfg["imputer_categorical_strategy"],
                                 fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    column_tf = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerics),
            ("cat", cat_pipeline, categoricals),
        ],
        remainder="drop"
    )

    steps = []

    if cfg["missing_indicator"]:
        steps.append(("missing_ind", MissingIndicatorAdder()))

    steps.append(("preproc", column_tf))

    final_pipe = Pipeline(steps)

    meta = {
        "numerics": numerics,
        "categoricals": categoricals,
        "config": cfg
    }

    return final_pipe, meta


# -----------------------------
# Fit + Transform
# -----------------------------

def fit_preprocessor(df, target_col=None, config=None) -> PreprocessResult:

    df = ensure_pandas(df)
    pipe, meta = build_preprocessor(df, target_col, config)

    cfg = meta["config"]

    # Leakage
    leak_report = {}
    if target_col and target_col in df.columns:
        leak_report = detect_leakage(df, target_col, cfg)

    # Prepare X / y
    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X = df
        y = None

    pipe.fit(X, y)

    X_processed = pipe.transform(X)

    # numpy → pandas
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    processed_df = pd.DataFrame(X_processed)

    summary = {
        "input_shape": df.shape,
        "processed_shape": processed_df.shape,
        "leak_report": leak_report,
        "meta": meta
    }

    return PreprocessResult(
        pipeline=pipe,
        processed_df=processed_df,
        summary=summary
    )
