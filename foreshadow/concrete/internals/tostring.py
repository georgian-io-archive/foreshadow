"""To String."""

from foreshadow.base import BaseEstimator, TransformerMixin
from foreshadow.wrapper import pandas_wrap


@pandas_wrap
class ToString(BaseEstimator, TransformerMixin):
    """Convert passed in data to string format."""

    def fit(self, X, y=None):
        """Empty fit.

        Args:
            X: input observations
            y: input labels

        Returns:
            self

        """
        return self

    def transform(self, X, y=None):
        """Convert a column to string form.

        Args:
            X (:obj:`pandas.DataFrame`): X data
            y: input labels

        Returns:
            :obj:`pandas.DataFrame`: Transformed data

        """
        return X.astype("str")
