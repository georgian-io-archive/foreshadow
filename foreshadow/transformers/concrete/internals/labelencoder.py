"""FixedLabelEncoder."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder


class FixedLabelEncoder(BaseEstimator, TransformerMixin):
    """Fix LabelEncoder function signature to fit transformer standard."""

    def __init__(self):
        self.encoder = SklearnLabelEncoder()

    def fit(self, X, y=None):
        """Fit the LabelEncoder.

        Args:
            X: iterable
            y (optional): iterable

        Returns:
            self

        """
        self.encoder.fit(X)
        return self

    def transform(self, X):
        """Transform using the fit LabelEncoder.

        Args:
            X: iterable

        Returns:
            array-like

        """
        return self.encoder.transform(X)

    def fit_transform(self, X):
        """Fit and transform in one step.

        Args:
            X: iterable

        Returns:
            array-like: Transformed samples

        """
        return self.encoder.fit_transform(X)

    def inverse_transform(self, X):
        """Transform labels back to original encoding.

        Args:
            X: iterable

        Returns:
            iterable: Inverted transformed samples

        """
        return self.encoder.inverse_transform(X)
