import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox1p
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    """Transforms data using a BoxCox transformation"""

    def fit(self, X, y=None):
        X = check_array(X)
        min_ = np.nanmin(X)
        self.translate_ = -min_ if min_ < 0 else 0
        _, self.lambda_ = boxcox(X + 1 + self.translate_)
        return self

    def transform(self, X, y=None):
        X = check_array(X, copy=True)
        check_is_fitted(self, ["translate_", "lambda_"])
        X = boxcox(X + 1 + self.translate_, self.lambda_)
        return X

    def inverse_transform(self, X):
        X = check_array(X, copy=True)
        check_is_fitted(self, ["translate_", "lambda_"])
        X = inv_boxcox1p(X, self.lambda_) - self.translate_
        return X
