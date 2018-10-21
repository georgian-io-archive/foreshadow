"""
General intents defenitions
"""

import pandas as pd
import numpy as np

from .base import BaseIntent

from ..transformers.internals import DropFeature
from ..transformers.smart import SimpleImputer, MultiImputer, Scaler, Encoder


class GenericIntent(BaseIntent):
    """See base class.

    Serves as root of Intent tree. In the case that no other intent applies this
    intent will serve as a placeholder.

    """

    dtype = "str"
    """Matches to string dtypes (not implemented)"""

    children = ["NumericIntent", "CategoricalIntent"]
    """Matches to CategoricalIntent over NumericIntent"""

    single_pipeline = []
    """No transformers"""

    multi_pipeline = [("multi_impute", MultiImputer())]
    """Performs multi imputation over the entire DataFrame"""

    @classmethod
    def is_intent(cls, df):
        """Returns true by default such that a column must match this"""
        return True


class NumericIntent(GenericIntent):
    """See base class.

    Matches to features with numerical data.

    """

    dtype = "float"
    """Matches to float dtypes (not implemented)"""

    children = []
    """No children"""

    single_pipeline = [
        ("dropper", DropFeature()),
        ("simple_imputer", SimpleImputer()),
        ("scaler", Scaler()),
    ]
    """Performs imputation and scaling using Smart Transformers"""

    multi_pipeline = []
    """No multi pipeline"""

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
    """Matches to integer dtypes (not implemented)"""

    children = []
    """No children"""

    single_pipeline = [("dropper", DropFeature()), ("impute_encode", Encoder())]
    """Encodes the column automatically"""

    multi_pipeline = []
    """No multi pipeline"""

    @classmethod
    def is_intent(cls, df):
        """Returns true if the majority of data is categorical by uniqueness"""
        data = df.ix[:, 0]
        if not np.issubdtype(data.dtype, np.number):
            return True
        else:
            return (1.0 * data.nunique() / data.count()) < 0.2
