"""Numeric intent."""

import pandas as pd

from foreshadow.metrics import (
    MetricWrapper,
    is_numeric,
    is_string,
    num_valid,
    unique_heur,
)
from foreshadow.utils import get_outliers, standard_col_summary

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
        result = standard_col_summary(df)

        data_transformed = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        invalid_pct = (
            data_transformed.isnull().sum() * 100.0 / result["count"]
            - result["nan_percent"]
        )
        outliers = get_outliers(data_transformed, count=5).values.tolist()

        result.update(
            [
                ("invalid_percent", invalid_pct),
                ("mean", float(data_transformed.mean())),
                ("std", float(data_transformed.std())),
                ("min", float(data_transformed.min())),
                ("25%", float(data_transformed.quantile(0.25))),
                ("50%", float(data_transformed.quantile(0.5))),
                ("75%", float(data_transformed.quantile(0.75))),
                ("max", float(data_transformed.max())),
                ("5_outliers", outliers),
            ]
        )
        return result
