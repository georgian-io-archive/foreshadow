"""Base package for all intent definitions."""

from functools import partial

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from foreshadow.metrics.internals import (
    is_numeric,
    is_string,
    num_valid,
    unique_heur,
)


class BaseIntent(BaseEstimator, TransformerMixin):
    """Base for all intent definitions.

    For each intent subclass a class attribute called `confidence_computation`
    must be defined which is of the form::

       {
            metric_def: weight
       }

    """

    @classmethod
    def get_confidence(cls, X, y=None):
        """Determine the confidence for an intent match.

        Args:
            X: input DataFrame.
            y: response variable

        Returns:
            float: A confidence value bounded between 0.0 and 1.0

        """
        return sum(
            [
                metric_fn(X) * weight
                for metric_fn, weight in cls.confidence_computation.items()
            ]
        )


class Numeric(BaseIntent):
    """Defines a numeric column type."""

    confidence_computation = {
        num_valid: 0.25,
        partial(unique_heur, invert=True): 0.25,
        is_numeric: 0.25,
        partial(is_string, invert=True): 0.25,
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
        """Convert a column to a numeric form.

        Args:
            X: The input data
            y: The response variable

        Returns:
            A column with all rows converted to numbers.

        """
        return X.apply(pd.to_numeric, errors="coerce")


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
