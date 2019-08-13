"""DropFeature."""
import numpy as np
import pandas as pd
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from foreshadow.base import BaseEstimator, TransformerMixin
from foreshadow.wrapper import pandas_wrap


@pandas_wrap
class DropFeature(BaseEstimator, TransformerMixin):
    """Drop data if it doesn't pass a set threshold.

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
        """Fit input data and set drop condition using threshold.

        Args:
            X (:obj:`pandas.DataFrame`): Fit data
            y: input labels

        Returns:
            self

        """
        X = check_array(X, force_all_finite=False, dtype=None).ravel()
        ratio = np.count_nonzero(~pd.isnull(X)) / X.size

        self.drop_ = ratio < self.threshold
        return self

    def transform(self, X, y=None):
        """Remove column based on drop condition.

        Args:
            X (:obj:`pandas.DataFrame`): X data
            y: input labels

        Returns:
            :obj:`pandas.DataFrame`: Transformed data

        """
        X = check_array(X, force_all_finite=False, dtype=None, copy=True)
        check_is_fitted(self, "drop_")
        return X if not self.drop_ else np.array([])

    def inverse_transform(self, X):
        """Return empty inverse.

        Args:
            X: input observations

        Returns:
            empty inverse

        Raises:
            ValueError: if self.raise_on_inverse

        """
        if not self.raise_on_inverse:
            return np.array([])
        else:
            raise ValueError(
                "inverse_transform is not permitted on this"
                " DropFeature instance"
            )
