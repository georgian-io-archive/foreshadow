"""Smart Transformers

Transformers here will be accessible through the namespace
foreshadow.transformers.smart and will not be
wrapped or transformed. Only classes extending SmartTransformer should exist here.

"""

from copy import deepcopy

import numpy as np
import scipy.stats as ss
from sklearn.pipeline import Pipeline

from ..transformers.base import SmartTransformer
from ..transformers.internals import (
    BoxCox,
    FancyImputer,
    PrepareFinancial,
    ConvertFinancial,
)
from ..transformers.externals import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    HashingEncoder,
    LabelEncoder,
    OneHotEncoder,
)


class Scaler(SmartTransformer):
    """Automatically Scales Numerical Features

    Analyzes the distribution of the data. If the data is normally distributed,
    StandardScaler is used, if it is uniform, MinMaxScaler is used, and if neither
    distribution fits then a BoxCox transformation is applied and a RobustScaler
    is used.
    """

    def _get_transformer(self, X, y=None, p_val_cutoff=0.05, **fit_params):
        data = X.iloc[:, 0]
        # statistically invalid but good enough measure of relative closeness
        # ks-test does not allow estimated parameters
        distributions = {"norm": StandardScaler(), "uniform": MinMaxScaler()}
        p_vals = {}
        for d in distributions.keys():
            dist = getattr(ss.distributions, d)
            p_vals[d] = ss.kstest(data, d, args=dist.fit(data)).pvalue
        best_dist = max(p_vals, key=p_vals.get)
        best_dist = best_dist if p_vals[best_dist] >= p_val_cutoff else None
        if best_dist is None:
            return Pipeline([("box_cox", BoxCox()), ("robust_scaler", RobustScaler())])
        else:
            return distributions[best_dist]


class Encoder(SmartTransformer):
    """Automatically Encodes Categorical Features

    If there are less than 30 categories, then OneHotEncoder is used, if there are more
    then HashingEncoder is used. If used in a y_var context, LabelEncoder is used.

    """

    def _get_transformer(self, X, y=None, unique_num_cutoff=30, **fit_params):
        data = X.iloc[:, 0]
        col_name = X.columns[0]
        unique_count = len(data.value_counts())
        if self.y_var:
            return LabelEncoder()
        if unique_count <= unique_num_cutoff:
            return OneHotEncoder(
                cols=[col_name],
                return_df=True,
                use_cat_names=True,
                handle_unknown="ignore",
            )
        else:
            return HashingEncoder(n_components=30, cols=[col_name])


class SimpleImputer(SmartTransformer):
    """Automatically Imputes Single Columns

    Performs z-score test to determine whether to use mean or median imputation. If
    too many data points are missing then imputation is not attempted in favor of
    multiple imputation later in the pipeline.

    """

    def __init__(self, threshold=0.1, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def _choose_simple(self, X):
        X = X[~np.isnan(X)]

        # Uses modified z score method http://colingorrie.github.io/outlier-detection.html
        # Assumes data is has standard distribution
        threshold = 3.5

        med_y = np.median(X)
        mad_y = np.median(np.abs(np.subtract(X, med_y)))
        z_scor = [0.6745 * (y - med_y) / mad_y for y in X]

        z_bool = np.where(np.abs(z_scor) > threshold)[0].shape[0] / X.shape[0] > 0.05

        if z_bool:
            return FancyImputer("SimpleFill", fill_method="median")
        else:
            return FancyImputer("SimpleFill", fill_method="mean")

    def _get_transformer(self, X, y=None, **fit_params):
        s = X.ix[:, 0]
        ratio = s.isnull().sum() / s.count()

        if 0 < ratio <= self.threshold:
            return self._choose_simple(s.values)
        else:
            return Pipeline([("null", None)])


class MultiImputer(SmartTransformer):
    """Automatically chooses a method of Multiple Imputation if neccesary

    By default, currently uses KNN multiple imputation as it is the fastest, and most
    flexible.

    """

    def _choose_multi(self, X):
        # For now simply default to KNN multiple imputation (generic case)
        # The rest of them seem to have constraints and no published directly comparable
        # performance

        # Impute using KNN
        return FancyImputer("KNN", k=3)

    def _get_transformer(self, X, y=None, **fit_params):
        if X.isnull().values.any():
            return self._choose_multi(X)
        else:
            return Pipeline([("null", None)])


class FinancialCleaner(SmartTransformer):
    """Automatically choose apropriate parameters for a financial column"""

    def _get_transformer(self, X, y=None, **fit_params):
        # number_regex = r'(?<!\S)(\[|\()?(((-(?=[0-9]))|(-\.(?=[0-9])))?[0-9]*((\.(?=[0-9]))|((?<=[0-9]))\.)?[0-9]*)(\)|\])?(?!\S)'

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
