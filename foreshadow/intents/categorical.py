"""Categorical intent."""

from foreshadow.metrics import num_valid, unique_heur, MetricWrapper2

from .base import BaseIntent


# Due to pickling issue in Parallel process, the lambda x: 1 needs to be
# rewritten as a public method.
def return_one(X):
    return 1


class Categoric(BaseIntent):
    """Defines a categoric column type."""

    confidence_computation = {
        MetricWrapper2(num_valid): (1 / 3),
        MetricWrapper2(unique_heur): (1 / 3),
        MetricWrapper2(return_one): (1 / 3),
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
