"""Numeric intent."""

from collections import OrderedDict

import pandas as pd

from foreshadow.metrics import (
    MetricWrapper,
    is_numeric,
    is_string,
    num_valid,
    unique_heur,
)
from foreshadow.utils import get_outliers, mode_freq

from .base import BaseIntent


class Numeric(BaseIntent):
    """Defines a numeric column type."""

    confidence_computation = {
        MetricWrapper(num_valid): 0.3,
        MetricWrapper(unique_heur, invert=True): 0.2,
        MetricWrapper(is_numeric): 0.4,
        MetricWrapper(is_string, invert=True): 0.1,
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

    @classmethod
    def column_summary(cls, df):  # noqa
        data = df.iloc[:, 0]
        nan_num = int(data.isnull().sum())
        invalid_num = int(
            pd.to_numeric(df.iloc[:, 0], errors="coerce").isnull().sum()
            - nan_num
        )
        outliers = get_outliers(data).values.tolist()
        mode, top10 = mode_freq(data)

        return OrderedDict(
            [
                ("nan", nan_num),
                ("invalid", invalid_num),
                ("mean", float(data.mean())),
                ("std", float(data.std())),
                ("min", float(data.min())),
                ("25th", float(data.quantile(0.25))),
                ("median", float(data.quantile(0.5))),
                ("75th", float(data.quantile(0.75))),
                ("max", float(data.max())),
                ("mode", mode),
                ("top10", top10),
                ("10outliers", outliers),
            ]
        )
