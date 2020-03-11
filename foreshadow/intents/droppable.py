"""Droppable intent. It does nothing."""

from foreshadow.intents import BaseIntent


class Droppable(BaseIntent):
    """Defines a droppable column type."""

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
        """Do nothing but return the original data frame.

        Args:
            X: The input data
            y: The response variable

        Returns:
            The original data frame

        """
        return X

    @classmethod
    def column_summary(cls, df):  # noqa
        raise NotImplementedError(
            "Column Summary is not implemented by " "the Droppable class."
        )
