import re

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class PrepareFinancial(BaseEstimator, TransformerMixin):
    """Cleans data in preparation for a financial transformer 
    (requires pandas inputs)
    """

    def fit(self, X, y=None):
        """Empty fit"""
        return self

    def transform(self, X, y=None):
        """Cleans string columns to prepare for financial transformer

        Args:
            X (:obj:`pandas.DataFrame`): X data

        Returns:
            :obj:`pandas.DataFrame`: Transformed data

        """

        X = X.copy()
        for c in X:
            X[c] = (
                X[c]
                .str.replace(r"\s", "")  # remove all whitespace
                .str.findall(r"[\d\.\(\[\-\)\]\,]+")  # keep valid characters
                .apply(
                    lambda l: max(l, key=len)  # match largest found group
                    if isinstance(l, list) and len(l) > 0
                    else np.nan
                )
            )

        return X


class ConvertFinancial(BaseEstimator, TransformerMixin):
    """Converts clean financial data into a numeric format

        Args:
            is_euro (bool): transform as a european number
    """

    def __init__(self, is_euro=False):
        self.is_euro = is_euro
        self.clean_us = r"(?<!\S)(\[|\()?((-(?=[0-9\.]))?([0-9](\,(?=[0-9]{3}))?)*((\.(?=[0-9]))|((?<=[0-9]))\.)?[0-9]*)(\)|\])?(?!\S)"

    def fit(self, X, y=None):
        """Empty fit"""
        return self

    def transform(self, X, y=None):
        """Prepares data to be processed by FinancialIntent

        Args:
            X (:obj:`pandas.DataFrame`): X data

        Returns:
            :obj:`pandas.DataFrame`: Transformed data

        """

        def get_match_results(val):
            if isinstance(val, str):
                match = re.compile(self.clean_us).match(val)
                if match:
                    return match.group()

            return np.nan

        X = X.copy()
        for c in X:
            if self.is_euro:
                X[c] = X[c].str.translate(str.maketrans(",.", ".,"))

            # Filter for validity
            X[c] = X[c].apply(get_match_results)

            X[c] = pd.to_numeric(
                X[c]
                .str.replace(r"[\(\[]", "-")  # accounting to negative
                .str.replace(r"[\]\)]", "")
                .str.replace(",", ""),  # remove thousand separator
                errors="coerce",  # convert to number
            )

        return X
