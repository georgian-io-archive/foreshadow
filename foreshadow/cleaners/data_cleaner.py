"""Cleaner module for cleaning data as step in Foreshadow workflow."""
from sklearn.base import BaseEstimator, TransformerMixin


class DataCleaner(BaseEstimator, TransformerMixin):
    """Wrapper class to determine and perform best data cleaning step."""

    def __init__(self, **kwargs):
        """Stub init method.

        Args:
            **kwargs: placeholder.

        """
        super().__init__()

    def fit(self, X, y=None, **fit_params):
        """Stub fit method.

        Args:
            X: input data
            y: labels
            **fit_params: params to fit method.

        """
        pass
