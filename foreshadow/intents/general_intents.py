"""
General intents defenitions
"""

from functools import partial

import pandas as pd
import numpy as np
import numpy.ma as ma
from sklearn.base import TransformerMixin

from .intents_base import BaseIntent

# from transforms import Binarizer, Imputer, RobustScaler, StandardScaler, OneHo #ToNumeric,


class GenericIntent(BaseIntent):
    intent = "generic"
    dtype = "str"
    children = ["NumericIntent"]

    def __init__(self, df, single_pipeline=True):
        if single_pipeline == True:
            raise TypeError("GenericIntent cannot be applied as a single pipeline")
        self.data = df.ix[:, 0] if single_pipeline else df
        self.temp_data = self.data.copy()

    @classmethod
    def has_outliers(cls, data):
        """pandas series input"""
        outlier_detector = lambda x, mu, std: np.abs((x - mu) / std) > 3.
        masked_data = ma.array(data.values, mask=data[~np.isnan(data)])
        pod = np.vectorize(
            partial(outlier_detector, mu=np.mean(masked_data), std=np.std(masked_data))
        )
        return np.any(pod(masked_data))

    @classmethod
    def is_intent(cls, df):
        return True

    def _internal_transform(self, encoder):
        self.temp_data = encoder.fit_transform(self.temp_data)

    def get_best_single_pipeline(self):
        raise TypeError("Cannot get single best pipeline for a GenericIntent")

    def adv_impute(self):
        # ignore columns with imputation steps and select only columns with
        # remaining missing values to impute using multiple imputation
        # assumes that all passed in values are gaussian
        # or do some other imputation method using some other heuristic
        enc = None
        # self._internal_transform(enc)
        return enc

    def boruta(self):
        # remove columns using boruta all relavent
        enc = None
        # self._internal_transform(enc)
        return enc

    def dim_red(self):
        # reduce dimensionality of large set of features
        enc = None
        # self._internal_transform(enc)
        return enc

    def get_best_multi_pipeline(self):
        steps = [
            ("adv_impute", self.adv_impute),
            ("boruta", self.boruta)("dim_red", self.dim_red),
        ]
        pipeline = []
        for s, f in steps:
            pipeline += (s, f())

        return [s for s in pipeline if not s[1] is None]


class NumericIntent(GenericIntent):
    intent = "numeric"
    dtype = "float"
    children = []

    def __init__(self, df, single_pipeline=True):
        self.data = df.ix[:, 0] if single_pipeline else df
        self.temp_data = self.data.copy()

    @classmethod
    def is_intent(cls, df):
        numeric_data = pd.to_numeric(df.ix[:, 0], errors="coerce")
        return (s.isnull().sum() / len(numeric_data)) > 0.5

    def impute(self):
        # Assume data is missing at random
        if not self.data.isnull().values.any():
            return None
        method = "mean" if not has_outliers(self.data) else "median"
        enc = Imputer(strategy=method)
        self._internal_transform(enc)
        return enc

    def scale(self):
        enc = StandardScaler() if not has_outliers(self.data) else RobustScaler()
        self._internal_transform(enc)
        return enc

    def get_best_single_pipeline(self):
        # go through each step in pipeline spec and optimize the transformer's
        # params
        steps = [("impute", self.impute), ("scale", self.scale)]
        pipeline = []
        for s, f in steps:
            pipeline += (s, f())

        return [s for s in pipeline if not s[1] is None]

    def pairwise(self):
        # TODO: generate pairwise features from an input dataframe set
        return None

    def get_best_multi_pipeline(self):
        steps = [("pairwise", self.pairwise)]
        pipeline = []
        for s, f in steps:
            pipeline += (s, f())

        return [s for s in pipeline if not s[1] is None]


class CategoricalIntent(GenericIntent):
    intent = "numeric"
    dtype = "int"
    children = []

    def __init__(self, df, single_pipeline=True):
        self.data = df.ix[:, 0] if single_pipeline else df

    @classmethod
    def is_intent(cls, df):
        """input pandas dataframe with single column"""
        data = df.ix[:, 0]
        if data.dtype != np.number:
            return True
        else:
            if cls.check_outliers(data):
                # check if the top 10 values account for a large percentage of feature
                return 1. * data.value_counts(normalize=True).head(10) > 0.8
            else:
                # check if the ratio of unique to total count is less than 5 percent
                return 1. * data.nunique() / data.count() < 0.05

    def impute(self):
        # TODO: Add imputer that accepts strings once transformer is complete
        enc = None
        # self._internal_transform(enc)
        return enc

    def to_ordinal(self):
        # TODO: return transformer that makes string categoric variables numeric
        enc = None
        # self._internal_transform(enc)
        return enc

    def bucketize(self):
        # TODO: If there are too many categories, group small values into a single category
        enc = None
        # self._internal_transform(enc)
        return enc

    def encoder(self):
        # TODO: Choose appropriate encoding strategy based on the input ordinal space
        enc = None
        # self._internal_transform(enc)
        return enc

    def get_best_single_pipeline(self):
        # go through each step in pipeline spec and optimize the transformer's
        # params
        steps = [
            ("numeric", self.to_ordinal),
            ("bucketize", self.bucketize),
            ("impute", self.impute)("encoder", self.encode),
        ]
        pipeline = []
        for s, f in steps:
            pipeline += (s, f())

        return [s for s in pipeline if not s[1] is None]

    def get_best_multi_pipeline(self):
        return None
