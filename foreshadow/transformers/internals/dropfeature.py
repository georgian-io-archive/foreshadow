import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class DropFeature(BaseEstimator, TransformerMixin):
    """Transformer that returns an emtpy array if the data doesn't pass a set
       threshold

        Parameters:
            threshold (float): if percentage of valid data is less than the
                threshold then the feature will be dropped
            raise_on_inverse (bool): allow or disallow return empty array on
                inverse

    """

    def __init__(self, threshold=0.3, raise_on_inverse=False, **kwargs):
        self.threshold = threshold
        self.raise_on_inverse = raise_on_inverse

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

    def inverse_transform(self, X):
        """Returns empty inverse"""
        if not self.raise_on_inverse:
            return np.array([])
        else:
            raise ValueError(
                "inverse_transform is not permitted on this"
                " DropFeature instance"
            )
