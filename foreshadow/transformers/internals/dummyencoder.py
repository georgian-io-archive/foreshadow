import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class DummyEncoder(BaseEstimator, TransformerMixin):
    """Dummy encodes delimmited data within column of dataframe"""

    def __init__(self, delimeter=","):
        self.delimeter = delimeter

    def fit(self, X, y=None):
        """Determines dummy categories

        Args:
            X (:obj:`numpy.ndarray`): Fit data

        Returns:
            self

        """
        X = X.iloc[:, 0]
        X = X.str.get_dummies(sep=self.delimeter)
        self.categories = list(X)

        return self

    def transform(self, X, y=None):
        """Performs Dummy Encoding on data

        Args:
            X (:obj:`numpy.ndarray`): X data

        Returns:
            :obj:`numpy.ndarray`: Transformed data

        """

        check_is_fitted(self, ["categories"])

        kwargs = {
            k: X.applymap(separate(k, self.delimeter)).iloc[:, 0].tolist()
            for k in self.categories
        }
        df = pd.DataFrame(kwargs)

        return df


def separate(cat, delim):
    def sep(X):
        if cat in X.split(delim):
            return 1
        return 0

    return sep
