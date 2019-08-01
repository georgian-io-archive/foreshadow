"""Text intent."""

from functools import partial
from .base import BaseIntent
from foreshadow.metrics import (
    num_valid,
    unique_heur,
    is_numeric,
    is_string,
)


class Text(BaseIntent):
    """Defines a text column type."""

    confidence_computation = {
        num_valid: 0.25,
        unique_heur: 0.25,
        partial(is_numeric, invert=True): 0.25,
        is_string: 0.25,
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
        """Convert a column to a text form.
        Args:
            X: The input data
            y: The response variable
        Returns:
            A column with all rows converted to text.
        """
        return X.astype(str)
