"""Smart Transformers.

Transformers here will be accessible through the namespace
:mod:`foreshadow.transformers.smart` and will not be wrapped or transformed.
Only classes extending SmartTransformer should exist here.

"""

from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.pipeline import Pipeline

from foreshadow.concrete import (
    NaNFiller,
    NoTransform,
    PowerTransformer,
    SimpleImputer,
)
from foreshadow.concrete.externals import (
    HashingEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
    TfidfVectorizer,
)
from foreshadow.concrete.internals import (
    ConvertFinancial,
    FancyImputer,
    FixedLabelEncoder as LabelEncoder,
    HTMLRemover,
    PrepareFinancial,
    ToString,
    UncommonRemover,
)
from foreshadow.logging import logging
from foreshadow.utils import (
    AcceptedKey,
    DataSeriesSelector,
    DefaultConfig,
    TruncatedSVDWrapper,
    check_df,
)

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
            selected_transformer = Pipeline(
                [
                    # Turning off the BoxCox transformer because if the test
                    # dataset has an even smaller negative min, it will
                    # break the pipeline.
                    # TODO add a different transformer if necessary
                    # ("box_cox", BoxCox()),
                    ("power_transformer", PowerTransformer()),
                    ("robust_scaler", RobustScaler()),
                ]
            )
        else:
            selected_transformer = distributions[best_dist]
        return selected_transformer


def will_remove_uncommon(X, temp_uncommon_remover):
    """Check if the transformer will modify the data.

    Uses current settings.

    Args:
        X: input observations column
        temp_uncommon_remover: transformer

    Returns:
        (tuple) bool and category counts

    """
    X = check_df(X, single_column=True).iloc[:, 0].values
    out = temp_uncommon_remover.fit_transform(X).values.ravel()

    return (
        not (np.array_equal(X, out) | (pd.isnull(X) & pd.isnull(out))).all(),
        pd.unique(out).size,
    )


class CategoricalEncoder(SmartTransformer):
    """Automatically encode categorical features.

    If there are no more than 30 categories, then OneHotEncoder is used,
    if there are more then HashingEncoder is used. If the columns containing a
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
        # NaN is treated as a separate category. In order to take it into
        # account during the econder selection, we fill the na value with
        # the string "NaN". In the final pipeline, it has a pre-defined
        # filler as the first step, which will take effect during the real
        # transformation.
        X = X.fillna("NaN")
        data = X.iloc[:, 0]
        unique_count = len(data.value_counts())

        # TODO Decided to temporarily turn off the DummyEncoder calculation.
        #  First of all, it is potentially very inefficient to apply the
        #  following logic on some non-multi-categorical data, especially
        #  when the data volume is large. One solution might be to check the
        #  presence of those delimeters first instead of directly applying.
        #  A better way is to ask the auto intent resolving step to produce
        #  a multi-categorical intent and deal with it separately. Second,
        #  even if the code decides to use DummyEncoder, the encoder's
        #  current implementation only creates category based on one particular
        #  delimiter. Then what's the point of check 4 types of delimiters
        #  here? This whole logic is flawed and we should remove this
        #  feature until we have a sound solution. For now, we should just
        #  state we do not support multi-categorical data.
        # Calculate stats for DummyEncoder
        # delimeters = [",", ";", "\t"]
        # delim_count = [
        #     len(list(data.astype("str").str.get_dummies(sep=d)))
        #     for d in delimeters
        # ]
        # delim_diff = min(delim_count) - len(list(pd.get_dummies(data)))

        # Calculate stats for UncommonRemover
        temp_uncommon_remover = UncommonRemover(threshold=self.merge_thresh)
        will_reduce, potential_reduced_count = will_remove_uncommon(
            X, temp_uncommon_remover
        )

        ohe = OneHotEncoder(
            return_df=True, use_cat_names=True, handle_unknown="ignore"
        )

        final_pipeline = Pipeline([("fill_na", NaNFiller(fill_value="NaN"))])

        if self.y_var:
            return LabelEncoder()
        # elif delim_diff < 0:
        #     delim = delimeters[delim_count.index(min(delim_count))]
        #     final_pipeline.steps.append(
        #         ("dummy_encodeer", DummyEncoder(delimeter=delim))
        #     )
        elif unique_count <= self.unique_num_cutoff:
            final_pipeline.steps.append(("one_hot_encoder", ohe))
        elif (
            potential_reduced_count <= self.unique_num_cutoff
        ) and will_reduce:
            final_pipeline.steps.append(
                (
                    "uncommon_remover",
                    UncommonRemover(threshold=self.merge_thresh),
                )
            )
            final_pipeline.steps.append(("one_hot_encoder", ohe))
        else:
            final_pipeline.steps.append(
                ("hash_encoder", HashingEncoder(n_components=30))
            )

        return final_pipeline


class SimpleFillImputer(SmartTransformer):
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
            return SimpleImputer()


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
            return Pipeline([("null", None)])


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
        us_pipeline = Pipeline(
            [("prepare", PrepareFinancial()), ("convert", ConvertFinancial())]
        )
        eu_pipeline = Pipeline(
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

    def __init__(
        self,
        n_components=DefaultConfig.N_COMPONENTS_SVD,
        html_cutoff=0.4,
        **kwargs
    ):
        self.html_cutoff = html_cutoff
        self.n_components = n_components

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
        steps = [
            (
                "data_series_selector",
                DataSeriesSelector(column_name=X.columns[0]),
            )
        ]

        # TODO Scheduled Remove. This is commented out because data with text
        #  intent is already converted into str type. As for html remover, I
        #  consider it as an optional feature. For now, it may be better that
        #  we only support pure text data until we have demands for handling
        #  raw html data. It uses regex so its performance is not very good.
        # data = X.iloc[:, 0]
        # if (data.dtype.type is not np.str_) and not all(
        #     [isinstance(i, str) for i in data]
        # ):
        #     steps.append(("num", ToString()))
        #
        # html_ratio = (
        #     data.astype("str").apply(HTMLRemover.is_html).sum()
        # ) / len(data)
        # if html_ratio > self.html_cutoff:
        #     steps.append(("hr", HTMLRemover()))

        # TODO: find heuristic for finding optimal values for values
        tfidf = TfidfVectorizer(
            decode_error="replace",
            strip_accents="unicode",
            min_df=0,
            sublinear_tf=True,
        )
        steps.append(("tfidf", tfidf))
        steps.append(
            (
                "truncated_svd",
                TruncatedSVDWrapper(
                    n_components=self.n_components, random_state=42
                ),
            )
        )

        return Pipeline(steps)

    def fit(self, X, y=None, **fit_params):  # noqa
        try:
            super().fit(X)
        except ValueError as e:
            if "empty vocabulary" in str(e):
                logging.error(
                    "The column {} may have wrong Intent type {}.".format(
                        X.columns[0],
                        self.cache_manager[AcceptedKey.INTENT, X.columns[0]],
                    )
                )
            raise e
        return self

    def transform(self, X):  # noqa
        Xt = super().transform(X=X)

        # Note that even if we specify the number of components we want, we may
        # get fewer components so we need to depend on the shape of Xt instead
        # of the n_components attributes.
        columns = [
            "svd_components_from_tfidf_" + str(i) for i in range(Xt.shape[1])
        ]

        # Here we need to make sure the index of the data frame is set to the
        # original. Otherwise, we will encounter data frame misalignment again.
        return pd.DataFrame(data=Xt, columns=columns, index=X.index)


class NeitherProcessor(SmartTransformer):
    """A temporary no transform processor for the Neither intent."""

    def __init__(self, html_cutoff=0.4, **kwargs):
        self.html_cutoff = html_cutoff

        super().__init__(**kwargs)

    def pick_transformer(self, X, y=None, **fit_params):
        """Determine the appropriate preprocessing method for Neither intent.

        Args:
            X (:obj:`pandas.DataFrame`): Input X data
            y (:obj: 'pandas.DataFrame'): labels Y for data
            **fit_params (dict): Parameters to apply to transformers when
                fitting

        Returns:
            A NoTransformer

        """
        return self._pick_transformer(X, y, **fit_params)

    def _pick_transformer(self, X, y=None, **fit_params):
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
            transformer = tfidf
        else:
            transformer = Pipeline(steps)

        return self._can_fit(transformer, X)

    def _can_fit(self, transformer, X, y=None, sampling_threshold=0.1):
        """Check if the TFIDF can be fitted on the sampled data.

        If not, it will default back to NoTransform.
        TODO: At this moment TFIDF is broken so it always default back to
         NoTransform.

        Args:
            transformer: selected transformer with TFIDF vectorizor
            X: the data frame
            y: the y variable data frame
            sampling_threshold: the threshold of the sampling

        Returns:
            Either the original transformer or the NoTransform

        """
        if len(X) * sampling_threshold < 30:
            # the rule of 30 to be statistically significant
            sampling_threshold = 1
        sample_df = X.sample(
            frac=sampling_threshold, replace=True, random_state=1
        )
        try:
            transformer.fit(sample_df)
            return transformer
        except Exception:
            # TODO change to ValueError once TFIDF is fixed.
            # logging.warning("Error during fit: ".format(str(e)))
            logging.warning(
                "Revert to NoTransform for Neither type temporarily."
            )
            return NoTransform()
