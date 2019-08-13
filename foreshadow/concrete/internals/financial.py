"""StandardDollarFinancial transformers."""

import re

import numpy as np
import pandas as pd

from foreshadow.base import BaseEstimator, TransformerMixin
from foreshadow.wrapper import pandas_wrap


@pandas_wrap
class PrepareFinancial(BaseEstimator, TransformerMixin):
    """Clean data in preparation for a financial transformer.

    Note: requires pandas input dataframes.

    """

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
        """Clean string columns to prepare for financial transformer.

        Args:
            X (:obj:`pandas.DataFrame`): X data
            y: input labels

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
    """Convert clean financial data into a numeric format.

    Args:
        is_euro (bool): transform as a european number

    """

    def __init__(self, is_euro=False):
        self.is_euro = is_euro
        self.clean_us = (
            r"(?<!\S)"  # Negative lookbehind any non whitespace
            r"(\[|\()?"  # Look for zero or 1 --> [ or (
            r"("  # CG 1:
            r"(-(?=[0-9\.]))?"  # Look for or or 1 (negative num case)
            r"([0-9](\,(?=[0-9]{3}))?)*"  # Positive num case w/ ,
            r"((\.(?=[0-9]))|((?<=[0-9]))\.)?[0-9]*)"  # decimals
            r"(\)|\])?"  # Look for zero or 1 --> ] or )
            r"(?!\S)"  # Negative lookahead whitespace
        )

    def fit(self, X, y=None):
        """Empty fit.

        Args:
            X: input observations
            y: input labels

        Returns:
            self

        """
        return self

    def transform(self, X, y=None):  # noqa: D202
        """Prepare data to be processed by FinancialIntent.

        Args:
            X (:obj:`pandas.DataFrame`): X data
            y: input labels

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
