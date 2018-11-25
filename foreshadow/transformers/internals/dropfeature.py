import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class DropFeature(BaseEstimator, TransformerMixin):
    """Transformer that returns an emtpy array if the data doesn't pass a set
       threshold"""

    def __init__(self, threshold=0.3, **kwargs):
        self.threshold = threshold

    def fit(self, X, y=None):
        """Fits input data and sets drop condition using initialized
           threshold value.

        Args:
            X (:obj:`pandas.DataFrame`): Fit data

        Returns:
            self

        """
        X = check_array(X, force_all_finite=False, dtype=None).ravel()
        ratio = np.count_nonzero(~pd.isnull(X)) / X.size

        self.drop_ = ratio < self.threshold
        return self

    def transform(self, X, y=None):
        """Drops column based on drop condition

        Args:
            X (:obj:`pandas.DataFrame`): X data

        Returns:
            :obj:`pandas.DataFrame`: Transformed data

        """
        X = check_array(X, force_all_finite=False, dtype=None, copy=True)
        check_is_fitted(self, "drop_")
        return X if not self.drop_ else np.array([])
