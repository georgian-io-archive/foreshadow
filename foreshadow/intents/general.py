"""
General intents defenitions
"""
import json
from collections import OrderedDict

import pandas as pd
import numpy as np

from .base import BaseIntent, PipelineTemplateEntry, TransformerEntry

from ..transformers.internals import DropFeature
from ..transformers.smart import SimpleImputer, MultiImputer, Scaler, Encoder


def mode_freq(series, count=10):
    """Computes the mode and the most frequent values

        Args:
            series (pandas.Series): the series to analyze
            count (int): the n number of most frequent values

    """

    vc = series.value_counts()
    if series[~series.isnull()].nunique() == 1:
        return None, []
    else:
        mode = series.mode().values.tolist()
        if len(mode) == 1:
            mode = mode[0]
        return (mode, vc.nlargest(count).reset_index().values.tolist())


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

    @classmethod
    def column_summary(cls, df):
        """No statistics can be computed for a general column"""
        return {}


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

    @classmethod
    def column_summary(cls, df):
        """Returns computed statistics for a NumericIntent column

            The following are computed:
                nan: count of nans pass into dataset
                invalid: number of invalid values after converting to numeric
                mean: -
                std: -
                min: -
                25th: 25th percentile
                median: -
                75th: 75th percentile
                max: -
                mode: mode or np.nan if data is mostly unique
                top10: top 10 most frequent values or empty array if mostly unique
                    [(value, count),...,]
                10outliers: largest 10 outliers

        """
        data = df.ix[:, 0]
        nan_num = int(data.isnull().sum())
        invalid_num = int(
            pd.to_numeric(df.ix[:, 0], errors="coerce").isnull().sum() - nan_num
        )
        outliers = (
            data[np.abs(data - data.mean()) > (3 * data.std())]
            .nlargest(10)
            .values.tolist()
        )
        mode, top10 = mode_freq(data)

        return OrderedDict(
            [
                ("nan", nan_num),
                ("invalid", invalid_num),
                ("mean", data.mean()),
                ("std", data.std()),
                ("min", data.min()),
                ("25th", data.quantile(0.25)),
                ("median", data.quantile()),
                ("75th", data.quantile(0.75)),
                ("max", data.max()),
                ("mode", mode),
                ("top10", top10),
                ("10outliers", outliers),
            ]
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

    @classmethod
    def column_summary(cls, df):
        """Returns computed statistics for a CategoricalIntent column

            The following are computed:
                nan: count of nans pass into dataset
                mode: mode or np.nan if data is mostly unique
                top10: top 10 most frequent values or empty array if mostly unique
                    [(value, count),...,]
        """
        data = df.ix[:, 0]
        nan_num = int(data.isnull().sum())
        mode, top10 = mode_freq(data)

        return OrderedDict([("nan", nan_num), ("mode", mode), ("top10", top10)])
