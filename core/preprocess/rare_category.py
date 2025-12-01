from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class RareCategoryMerger(BaseEstimator, TransformerMixin):
    """
    Merge categories that are below frequency threshold into a single 'OTHER' bucket.
    """
    def __init__(self, threshold=0.01, fill_value="__OTHER__"):
        self.threshold = threshold
        self.fill_value = fill_value
        self.frequent_maps_ = {}

    def fit(self, X, y=None):
        # X expected to be DataFrame or 2D array-like with column names preserved
        df = pd.DataFrame(X)
        for col in df.columns:
            freqs = df[col].value_counts(normalize=True)
            keep = set(freqs[freqs >= self.threshold].index)
            self.frequent_maps_[col] = keep
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for col, keep in self.frequent_maps_.items():
            df[col] = df[col].apply(lambda v: v if v in keep else self.fill_value)
        return df
