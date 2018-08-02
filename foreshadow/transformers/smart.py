"""Smart Transformers

Transformers here will be accessible through the namespace
foreshadow.transformers.smart OR simple foreshadow.transformers and will not be
wrapped or transformed. Only classes extending SmartTransformer should exist here.

"""

from .imputer import SimpleImputer, MultiImputer

from ..transformers import SmartTransformer
from ..transformers import MinMaxScaler, StandardScaler, RobustScaler, BoxCoxTransformer
from ..transformers import OneHotEncoder, HashingEncoder

import scipy.stats as ss
from sklearn.pipeline import Pipeline


class SmartScaler(SmartTransformer):
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
            return Pipeline(
                [("box_cox", BoxCoxTransformer()), ("robust_scaler", RobustScaler())]
            )
        else:
            return distributions[best_dist]


class SmartCoder(SmartTransformer):
    def _get_transformer(self, X, y=None, unique_num_cutoff=30, **fit_params):
        data = X.iloc[:, 0]
        col_name = X.columns[0]
        unique_count = len(data.value_counts())
        if unique_count <= unique_num_cutoff:
            return OneHotEncoder(cols=[col_name])
        else:
            return HashingEncoder(n_components=30, cols=[col_name])
