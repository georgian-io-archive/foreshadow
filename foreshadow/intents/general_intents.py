"""
General intents defenitions
"""

import pandas as pd
import numpy as np

from .intents_base import BaseIntent
from ..transformers import Imputer, PCA


class GenericIntent(BaseIntent):
    """See base class.

    Serves as root of Intent tree. In the case that no other intent applies this
    intent will serve as a placeholder.

    """

    dtype = "str"
    children = ["NumericIntent", "CategoricalIntent"]

    multi_pipeline = [("pca", PCA(n_components=2))]
    single_pipeline = []

    @classmethod
    def is_intent(cls, df):
        """Returns true by default"""
        return True


class NumericIntent(GenericIntent):
    """See base class.

    Matches to features with numerical data.

    """

    dtype = "float"
    children = []

    single_pipeline = [("impute", Imputer(strategy="mean"))]
    multi_pipeline = []

    @classmethod
    def is_intent(cls, df):
        """Returns true if data is numeric according to pandas."""
        return True
        numeric_data = pd.to_numeric(df.ix[:, 0], errors="coerce")
        return (s.isnull().sum() / len(numeric_data)) > 0.5


class CategoricalIntent(GenericIntent):
    """See base class.

    Matches to features with low enough variance that encoding should be used.

    """

    dtype = "int"
    children = []

    @classmethod
    def is_intent(cls, df):
        """Returns true if the majority of data consists of the top 10 values."""
        return False
        data = df.ix[:, 0]
        if data.dtype != np.number:
            return True
        else:
            if cls.has_outliers(data):
                # check if the top 10 values account for a large percentage of feature
                return 1. * data.value_counts(normalize=True).head(10) > 0.8
            else:
                # check if the ratio of unique to total count is less than 5 percent
                return 1. * data.nunique() / data.count() < 0.05
