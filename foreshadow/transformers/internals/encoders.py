import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class OneHotEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        """No fit needed"""
        return self

    def transform(self, X):
        """Runs simple pd get_dummies."""
        df = pd.get_dummies(X)
        df.columns = [c.split("_")[1] for c in df.columns]
        return df
