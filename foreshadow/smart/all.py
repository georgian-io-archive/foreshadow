"""Smart Transformers.

Transformers here will be accessible through the namespace
:mod:`foreshadow.transformers.smart` and will not be wrapped or transformed.
Only classes extending SmartTransformer should exist here.

"""

from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.stats as ss

from foreshadow.concrete.externals import (
    HashingEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
    TfidfVectorizer,
)
from foreshadow.concrete.internals import (
    BoxCox,
    ConvertFinancial,
    DummyEncoder,
    FancyImputer,
    FixedLabelEncoder as LabelEncoder,
    HTMLRemover,
    PrepareFinancial,
    ToString,
    UncommonRemover,
)
from foreshadow.pipeline import SerializablePipeline
from foreshadow.utils import check_df

from .smart import SmartTransformer


# TODO: split this file up


class Scaler(SmartTransformer):
    """Automatically scale numerical features.

    Analyzes the distribution of the data. If the data is normally distributed,
    StandardScaler is used, if it is uniform, MinMaxScaler is used, and if
    neither distribution fits then a BoxCox transformation is applied and a
    RobustScaler is used.

    Args:
        p_val (float): p value cutoff for the ks-test

    """

    def __init__(self, p_val=0.05, **kwargs):
        self.p_val = p_val
        super().__init__(**kwargs)

    def pick_transformer(self, X, y=None, **fit_params):
        """Determine the appropriate scaling method for an input dataset.

        Args:
            X (:obj:`pandas.DataFrame`): Input X data
            y (:obj: 'pandas.DataFrame'): labels Y for data
            **fit_params (dict): Parameters to apply to transformers when
                fitting

        Returns:
            An initialized scaling transformer

        """
        X = check_df(X)
        data = X.iloc[:, 0]
        # statistically invalid but good enough measure of relative closeness
        # ks-test does not allow estimated parameters
        distributions = {"norm": StandardScaler(), "uniform": MinMaxScaler()}
        p_vals = {}
        for d in distributions.keys():
            dist = getattr(ss.distributions, d)
            p_vals[d] = ss.kstest(data, d, args=dist.fit(data)).pvalue
        best_dist = max(p_vals, key=p_vals.get)
        best_dist = best_dist if p_vals[best_dist] >= self.p_val else None
        if best_dist is None:
            return SerializablePipeline(
                [("box_cox", BoxCox()), ("robust_scaler", RobustScaler())]
            )
        else:
            return distributions[best_dist]


class CategoricalEncoder(SmartTransformer):
    """Automatically encode categorical features.

    If there are less than 30 categories, then OneHotEncoder is used, if there
    are more then HashingEncoder is used. If the columns containing a
    delimmeter exceed delim_cuttoff then a DummyEncoder is used (set cutoff to
    -1 to force). If used in a y_var context, LabelEncoder is used.

    Args:
        unique_num_cutoff (float): number of allowable unique categories
        merge_thresh (float): threshold passed into UncommonRemover if
            selected

    """

    def __init__(self, unique_num_cutoff=30, merge_thresh=0.01, **kwargs):
        self.unique_num_cutoff = unique_num_cutoff
        self.merge_thresh = merge_thresh
        super().__init__(**kwargs)

    def will_transform(self, X, temp_ur):
        """Check if the transformer will modify the data.

        Uses current settings.

        Args:
            X: input observations column
            temp_ur: transformer

        Returns:
            (tuple) bool and category counts

        """
        X = check_df(X, single_column=True).iloc[:, 0].values
        out = temp_ur.fit_transform(X).values.ravel()

        return (
            not (
                np.array_equal(X, out) | (pd.isnull(X) & pd.isnull(out))
            ).all(),
            pd.unique(out).size,
        )

    def pick_transformer(self, X, y=None, **fit_params):
        """Determine the appropriate encoding method for an input dataset.

        Args:
            X (:obj:`pandas.DataFrame`): Input X data
            y (:obj: 'pandas.DataFrame'): labels Y for data
            **fit_params (dict): Parameters to apply to transformers when
                fitting

        Returns:
            An initialized encoding transformer

        """
        data = X.iloc[:, 0]
        unique_count = len(data.value_counts())

        delimeters = [",", ";", "\t"]
        delim_count = [
            len(list(data.astype("str").str.get_dummies(sep=d)))
            for d in delimeters
        ]
        delim_diff = min(delim_count) - len(list(pd.get_dummies(data)))
        temp_ur = UncommonRemover(threshold=self.merge_thresh)
        will_reduce, reduce_count = self.will_transform(X, temp_ur)
        ohe = OneHotEncoder(
            return_df=True, use_cat_names=True, handle_unknown="ignore"
        )

        if self.y_var:
            return LabelEncoder()
        elif delim_diff < 0:
            delim = delimeters[delim_count.index(min(delim_count))]
            return DummyEncoder(delimeter=delim)
        elif unique_count <= self.unique_num_cutoff:
            return ohe
        elif (reduce_count <= self.unique_num_cutoff) and will_reduce:
            return SerializablePipeline(
                [
                    ("ur", UncommonRemover(threshold=self.merge_thresh)),
                    ("ohe", ohe),
                ]
            )
        else:
            return HashingEncoder(n_components=30)


class SimpleImputer(SmartTransformer):
    """Automatically impute single columns.

    Performs z-score test to determine whether to use mean or median
    imputation. If too many data points are missing then imputation is not
    attempted in favor of multiple imputation later in the pipeline.

        Args:
            threshold (float): threshold of missing data where to use these
                strategies
    """

    def __init__(self, threshold=0.1, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def _choose_simple(self, X):
        X = X[~np.isnan(X)]

        # Uses modified z score method
        # http://colingorrie.github.io/outlier-detection.html
        # Assumes data is has standard distribution
        z_threshold = 3.5

        med_y = np.median(X)
        mad_y = np.median(np.abs(np.subtract(X, med_y)))
        z_scor = [0.6745 * (y - med_y) / mad_y for y in X]

        z_bool = (
            np.where(np.abs(z_scor) > z_threshold)[0].shape[0] / X.shape[0]
            > 0.05
        )

        if z_bool:
            return FancyImputer(
                "SimpleFill", impute_kwargs={"fill_method": "median"}
            )
        else:
            return FancyImputer(
                "SimpleFill", impute_kwargs={"fill_method": "mean"}
            )

    def pick_transformer(self, X, y=None, **fit_params):
        """Determine the appropriate imputation method for an input dataset.

        Args:
            X (:obj:`pandas.DataFrame`): Input X data
            y (:obj: 'pandas.DataFrame'): labels Y for data
            **fit_params (dict): Parameters to apply to transformers when
                fitting

        Returns:
            An initialized imputation transformer

        """
        s = X.iloc[:, 0]
        ratio = s.isnull().sum() / s.count()

        if 0 < ratio <= self.threshold:
            return self._choose_simple(s.values)
        else:
            return SerializablePipeline([("null", None)])


class MultiImputer(SmartTransformer):
    """Automatically choose a method of multiple imputation.

    By default, currently uses KNN multiple imputation as it is the fastest,
    and most flexible.

    """

    def _choose_multi(self, X):
        # For now simply default to KNN multiple imputation (generic case)
        # The rest of them seem to have constraints and no published directly
        # comparable
        # performance

        # Impute using KNN
        return FancyImputer("KNN", impute_kwargs={"k": 3})

    def pick_transformer(self, X, y=None, **fit_params):
        """Determine the appropriate multiple imputation method.

        Args:
            X (:obj:`pandas.DataFrame`): Input X data
            y (:obj: 'pandas.DataFrame'): labels Y for data
            **fit_params (dict): Parameters to apply to transformers when
                fitting

        Returns:
            An initialized multiple imputation transformer

        """
        if X.isnull().values.any():
            return self._choose_multi(X)
        else:
            return SerializablePipeline([("null", None)])


class FinancialCleaner(SmartTransformer):
    """Automatically choose appropriate parameters for a financial column."""

    def pick_transformer(self, X, y=None, **fit_params):
        """Determine the appropriate financial cleaning method.

        Args:
            X (:obj:`pandas.DataFrame`): Input X data
            y (:obj: 'pandas.DataFrame'): labels Y for data
            **fit_params (dict): Parameters to apply to transformers when
                fitting

        Returns:
            An initialized financial cleaning transformer

        """
        us_pipeline = SerializablePipeline(
            [("prepare", PrepareFinancial()), ("convert", ConvertFinancial())]
        )
        eu_pipeline = SerializablePipeline(
            [
                ("prepare", PrepareFinancial()),
                ("convert", ConvertFinancial(is_euro=True)),
            ]
        )
        us_data = deepcopy(us_pipeline).fit_transform(X)
        eu_data = deepcopy(eu_pipeline).fit_transform(X)

        if eu_data.isnull().values.sum() < us_data.isnull().values.sum():
            return eu_pipeline
        else:
            return us_pipeline


class TextEncoder(SmartTransformer):
    """Automatically choose appropriate parameters for a text column.

    Args:
        threshold (float): threshold of missing data where to use these
            strategies
    """

    def __init__(self, html_cutoff=0.4, **kwargs):
        self.html_cutoff = html_cutoff

        super().__init__(**kwargs)

    def pick_transformer(self, X, y=None, **fit_params):
        """Determine the appropriate nlp method.

        Args:
            X (:obj:`pandas.DataFrame`): Input X data
            y (:obj: 'pandas.DataFrame'): labels Y for data
            **fit_params (dict): Parameters to apply to transformers when
                fitting

        Returns:
            An initialized nlp transformer

        """
        data = X.iloc[:, 0]

        steps = []

        if (data.dtype.type is not np.str_) and not all(
            [isinstance(i, str) for i in data]
        ):
            steps.append(("num", ToString()))

        html_ratio = (
            data.astype("str").apply(HTMLRemover.is_html).sum()
        ) / len(data)
        if html_ratio > self.html_cutoff:
            steps.append(("hr", HTMLRemover()))

        # TODO: find heuristic for finding optimal values for values
        tfidf = TfidfVectorizer(
            decode_error="replace",
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=0.05,
            max_features=None,
            sublinear_tf=True,
        )
        steps.append(("tfidf", tfidf))

        if len(steps) == 1:
            return tfidf
        else:
            return SerializablePipeline(steps)
