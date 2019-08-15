"""No Transform class through acts as a pass through for DataFrame and flag."""
from foreshadow.base import BaseEstimator, TransformerMixin
from foreshadow.wrapper import pandas_wrap


@pandas_wrap
class NoTransform(BaseEstimator, TransformerMixin):
    """Transformer that performs _Empty transformation."""

    def fit(self, X, y=None):
        """Empty fit function.

        Args:
            X (:obj:`numpy.ndarray`): input data to fit, observations
            y: labels

        Returns:
            self

        """
        return self

    def transform(self, X, y=None):
        """Pass through transform.

        Args:
            X (:obj:`numpy.ndarray`): X data
            y: labels

        Returns:
            :obj:`numpy.ndarray`: Empty numpy array

        """
        return X

    def inverse_transform(self, X):
        """Pass through transform.

        Args:
            X (:obj:`numpy.ndarray`): X data

        Returns:
            :obj:`numpy.ndarray`: Empty numpy array

        """
        return X
