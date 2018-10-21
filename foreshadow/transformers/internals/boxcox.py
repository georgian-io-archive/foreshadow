import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox1p
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class BoxCox(BaseEstimator, TransformerMixin):
    """Transformer that performs BoxCox transformation on continuous numeric data."""

    def fit(self, X, y=None):
        """Fits translate and lambda attributes to X data

        Args:
            X (:obj:`numpy.ndarray`): Fit data

        Returns:
            self

        """
        X = check_array(X)
        min_ = np.nanmin(X)
        self.translate_ = -min_ if min_ < 0 else 0
        _, self.lambda_ = boxcox(X + 1 + self.translate_)
        return self

    def transform(self, X, y=None):
        """Performs Box Cox transform on data

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
        """Reverses Box Cox transform

        Args:
            X (:obj:`numpy.ndarray`): Transformed X data

        Returns:
            :obj:`numpy.ndarray`: Original data

        """
        X = check_array(X, copy=True)
        check_is_fitted(self, ["translate_", "lambda_"])
        X = inv_boxcox1p(X, self.lambda_) - self.translate_
        return X
