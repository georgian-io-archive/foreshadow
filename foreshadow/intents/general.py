"""
General intents defenitions
"""

import pandas as pd
import numpy as np

from .base import BaseIntent, PipelineTemplateEntry, TransformerEntry

from ..transformers.internals import DropFeature
from ..transformers.smart import SimpleImputer, MultiImputer, Scaler, Encoder


class GenericIntent(BaseIntent):
    """See base class.

    Serves as root of Intent tree. In the case that no other intent applies this
    intent will serve as a placeholder.

    """

    children = ["NumericIntent", "CategoricalIntent"]
    """Matches to CategoricalIntent over NumericIntent"""

    single_pipeline_template = []
    """No transformers"""

    multi_pipeline_template = [
        PipelineTemplateEntry("multi_impute", MultiImputer, False)
    ]
    """Performs multi imputation over the entire DataFrame"""

    @classmethod
    def is_intent(cls, df):
        """Returns true by default such that a column must match this"""
        return True


class NumericIntent(GenericIntent):
    """See base class.

    Matches to features with numerical data.

    """

    children = []
    """No children"""

    single_pipeline_template = [
        PipelineTemplateEntry("dropper", DropFeature, False),
        PipelineTemplateEntry("simple_imputer", SimpleImputer, False),
        PipelineTemplateEntry("scaler", Scaler, True),
    ]
    """Performs imputation and scaling using Smart Transformers"""

    multi_pipeline_template = []
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

    children = []
    """No children"""

    single_pipeline_template = [
        PipelineTemplateEntry("dropper", DropFeature, False),
        PipelineTemplateEntry("impute_encode", Encoder, True),
    ]
    """Encodes the column automatically"""

    multi_pipeline_template = []
    """No multi pipeline"""

    @classmethod
    def is_intent(cls, df):
        """Returns true if the majority of data is categorical by uniqueness"""
        data = df.ix[:, 0]
        if not np.issubdtype(data.dtype, np.number):
            return True
        else:
            return (1.0 * data.nunique() / data.count()) < 0.2
