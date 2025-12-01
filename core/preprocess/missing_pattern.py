from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    """
    Add boolean columns is_<col>_missing for every input column (optionally only for subset).
    """
    def __init__(self, only_for=None):
        # only_for: list of columns or None (all)
        self.only_for = only_for

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        if self.only_for is None:
            self.cols_ = list(df.columns)
        else:
            self.cols_ = [c for c in self.only_for if c in df.columns]
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        out = df.copy()
        for c in self.cols_:
            out[f"isna__{c}"] = df[c].isna().astype(int)
        return out
