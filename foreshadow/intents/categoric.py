"""Categorical intent."""

from foreshadow.metrics import num_valid, unique_heur

from .base import BaseIntent


class Categoric(BaseIntent):
    """Defines a categoric column type."""

    confidence_computation = {
        num_valid: (1 / 3),
        unique_heur: (1 / 3),
        lambda x: 1: (1 / 3),
    }

    def fit(self, X, y=None, **fit_params):
        """Empty fit.

        Args:
            X: The input data
            y: The response variable
            **fit_params: Additional parameters for the fit

        Returns:
            self

        """
        return self

    def transform(self, X, y=None):
        """Pass-through transform.

        Args:
            X: The input data
            y: The response variable

        Returns:
            The input column

        """
        return X
