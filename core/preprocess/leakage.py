"""
Leakage detection utilities.
Two heuristics:
1) near-perfect correlation (numeric) or perfect mapping for categorical values.
2) mutual information (for classification/regression) - heuristic threshold.
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from typing import Dict

def _mutual_info(x, y, discrete_target=False):
    # Choose correct mutual info func depending on y dtype
    try:
        if discrete_target:
            return mutual_info_classif(x.reshape(-1,1), y, discrete_features='auto', random_state=0)[0]
        else:
            return mutual_info_regression(x.reshape(-1,1), y, discrete_features='auto', random_state=0)[0]
    except Exception:
        return 0.0

def detect_leakage(df: pd.DataFrame, target_col: str, config: Dict):
    report = { "leaks": [] }
    y = df[target_col].values
    discrete_target = pd.api.types.is_integer_dtype(df[target_col]) or pd.api.types.is_bool_dtype(df[target_col]) or pd.api.types.is_categorical_dtype(df[target_col]) or df[target_col].nunique() < 20

    for col in df.columns:
        if col == target_col:
            continue
        series = df[col]
        # numeric correlation test
        if pd.api.types.is_numeric_dtype(series):
            # handle constant or nan
            if series.nunique(dropna=True) <= 1:
                continue
            corr = series.corr(df[target_col])
            if pd.notna(corr) and abs(corr) >= config.get("leakage_corr_threshold", 0.95):
                report["leaks"].append({"column": col, "reason": "high_correlation", "value": float(corr)})
                continue
        # exact or near-mapping for categorical-like
        if series.nunique(dropna=True) < 50:
            # create mapping accuracy
            tmp = pd.DataFrame({"x": series.astype(str), "y": df[target_col].astype(str)})
            best_map = tmp.groupby("x")["y"].agg(lambda s: s.mode().iat[0] if len(s.mode())>0 else None)
            preds = tmp["x"].map(best_map)
            acc = (preds == tmp["y"]).mean()
            if acc >= 0.95:
                report["leaks"].append({"column": col, "reason": "almost_perfect_mapping", "value": float(acc)})
                continue
        # mutual information heuristic
        try:
            xi = series.fillna(-9999)
            if pd.api.types.is_numeric_dtype(series):
                mi = _mutual_info(xi.values.astype(float), y, discrete_target)
            else:
                le = LabelEncoder()
                xi_enc = le.fit_transform(xi.astype(str))
                mi = _mutual_info(xi_enc, y, discrete_target)
            if mi >= config.get("leakage_mi_threshold", 0.6):
                report["leaks"].append({"column": col, "reason": "high_mutual_info", "value": float(mi)})
        except Exception:
            continue

    return report
