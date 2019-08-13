"""BoxCox transform class."""

import numpy as np
from scipy.special import inv_boxcox1p
from scipy.stats import boxcox
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from foreshadow.base import BaseEstimator, TransformerMixin
from foreshadow.wrapper import pandas_wrap


@pandas_wrap
class BoxCox(BaseEstimator, TransformerMixin):
    """Perform BoxCox transformation on continuous numeric data."""

    # TODO: Remove this internal function and use PowerTransform from sklearn
    # when sklearn version is upgraded to 0.20

    def fit(self, X, y=None):
        """Fit translate and lambda attributes to X data.

        Args:
            X (:obj:`numpy.ndarray`): Fit data
            y: input labels

        Returns:
            self

        """
        X = check_array(X)
        min_ = np.nanmin(X)
        self.translate_ = -min_ if min_ <= 0 else 0
        _, self.lambda_ = boxcox(X + 1 + self.translate_)
        return self

    def transform(self, X):
        """Perform Box Cox transform on input.

        Args:
            X (:obj:`numpy.ndarray`): X data

        Returns:
            :obj:`numpy.ndarray`: Transformed data

        """
        X = check_array(X, copy=True)
        check_is_fitted(self, ["translate_", "lambda_"])
        X = boxcox(X + 1 + self.translate_, self.lambda_)
        return X

    def inverse_transform(self, X):
        """Reverse Box Cox transform.

        Args:
            X (:obj:`numpy.ndarray`): Transformed X data

        Returns:
            :obj:`numpy.ndarray`: Original data

        """
        X = check_array(X, copy=True)
        check_is_fitted(self, ["translate_", "lambda_"])
        X = np.clip(X, a_min=-0.99, a_max=None)
        X = inv_boxcox1p(X, self.lambda_) - self.translate_
        return X
