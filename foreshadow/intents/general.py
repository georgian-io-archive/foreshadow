"""
General intents defenitions
"""

import pandas as pd
import numpy as np

from .base import BaseIntent

from ..transformers.smart import SimpleImputer, MultiImputer, Scaler, Encoder


class GenericIntent(BaseIntent):
    """See base class.

    Serves as root of Intent tree. In the case that no other intent applies this
    intent will serve as a placeholder.

    """

    dtype = "str"
    children = ["NumericIntent", "CategoricalIntent"]

    single_pipeline = []
    multi_pipeline = [("multi_impute", MultiImputer())]

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

    single_pipeline = [("simple_imputer", SimpleImputer()), ("scaler", Scaler())]
    multi_pipeline = []

    @classmethod
    def is_intent(cls, df):
        """Returns true if data is numeric according to pandas."""
        return (
            not pd.to_numeric(df.ix[:, 0], errors="coerce")
            .isnull()
            .values.ravel()
            .all()
        )


class CategoricalIntent(GenericIntent):
    """See base class.

    Matches to features with low enough variance that encoding should be used.

    """

    dtype = "int"
    children = []

    single_pipeline = [("impute_encode", Encoder())]
    multi_pipeline = []

    @classmethod
    def is_intent(cls, df):
        """Returns true if the majority of data is categorical"""
        data = df.ix[:, 0]
        if not np.issubdtype(data.dtype, np.number):
            return True
        else:
            return (1. * data.nunique() / data.count()) < 0.2
