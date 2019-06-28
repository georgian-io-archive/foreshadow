"""FixedTfidfVectorizer."""

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import (
    TfidfVectorizer as SklearnTfidfVectorizer,
    VectorizerMixin,
)
from sklearn.utils import check_array


class FixedTfidfVectorizer(BaseEstimator, VectorizerMixin):
    """Fix TfidfVectorizer input format to fit transformer standard."""

    def __init__(self, **kwargs):
        self.encoder = SklearnTfidfVectorizer(**kwargs)

    def fit(self, X, y=None):
        """Fit the TfidfVectorizer.

        Args:
            X: iterable
            y (optional): iterable

        Returns:
            self

        """
        X = check_array(
            X, accept_sparse=True, dtype=None, force_all_finite=False
        ).ravel()
        self.encoder.fit(X, y)
        return self

    def transform(self, X):
        """Transform using the fit TfidfVectorizer.

        Args:
            X: iterable

        Returns:
            array-like

        """
        X = check_array(
            X, accept_sparse=True, dtype=None, force_all_finite=False
        ).ravel()
        return self.encoder.transform(X)

    def fit_transform(self, X, y=None):
        """Fit and transform in one step.

        Args:
            X: iterable

        Returns:
            array-like: Transformed samples

        """
        X = check_array(
            X, accept_sparse=True, dtype=None, force_all_finite=False
        ).ravel()
        return self.encoder.fit_transform(X, y)

    def inverse_transform(self, X):
        """Transform encoding back to original encoding.

        Args:
            X: iterable

        Returns:
            iterable: Inverted transformed samples

        """
        return [list(i) for i in self.encoder.inverse_transform(X)]
