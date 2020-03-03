"""Categorical intent."""

from foreshadow.metrics import (
    MetricWrapper,
    is_numeric,
    num_valid,
    unique_heur,
)
from foreshadow.utils import standard_col_summary

from .base import BaseIntent


class Categorical(BaseIntent):
    """Defines a categoric column type."""

    confidence_computation = {
        MetricWrapper(num_valid): 0.25,
        MetricWrapper(unique_heur): 0.65,
        MetricWrapper(is_numeric, invert=True): 0.1,
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

    @classmethod
    def column_summary(cls, df):  # noqa
        return standard_col_summary(df)
