"""Fill NaNs."""

from foreshadow.base import BaseEstimator, TransformerMixin
from foreshadow.utils import Constant
from foreshadow.wrapper import pandas_wrap


@pandas_wrap
class NaNFiller(BaseEstimator, TransformerMixin):
    """Fill NaN values in data."""

    def __init__(self, fill_value=Constant.NAN_FILL_VALUE):
        self.fill_value = fill_value

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
        """Fill nans in a column with defined fill_value.

        Args:
            X (:obj:`pandas.DataFrame`): X data
            y: input labels

        Returns:
            :obj:`pandas.DataFrame`: Transformed data

        """
        return X.fillna(self.fill_value)
