import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class ToString(BaseEstimator, TransformerMixin):
    """Converst passed in data to string format"""

    def fit(self, X, y=None):
        """Empty fit"""
        return self

    def transform(self, X, y=None):
        """Converts a column to string form

        Args:
            X (:obj:`pandas.DataFrame`): X data

        Returns:
            :obj:`pandas.DataFrame`: Transformed data

        """

        return X.astype("str")
