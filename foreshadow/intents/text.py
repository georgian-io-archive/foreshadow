"""Text intent."""

from foreshadow.metrics import (
    MetricWrapper,
    is_numeric,
    is_string,
    num_valid,
    unique_heur,
)

from .base import BaseIntent


class Text(BaseIntent):
    """Defines a text column type."""

    confidence_computation = {
        MetricWrapper(num_valid): 0.25,
        MetricWrapper(unique_heur): 0.25,
        MetricWrapper(is_numeric, invert=True): 0.25,
        MetricWrapper(is_string): 0.25,
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
