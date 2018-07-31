"""Smart Transformers

Transformers here will be accessible through the namespace
foreshadow.transformers.smart OR simple foreshadow.transformers and will not be
wrapped or transformed. Only classes extending SmartTransformer should exist here.

"""

from .imputer import SimpleImputer, MultiImputer

from ..transformers import SmartTransformer
from ..transformers import MinMaxScaler, StandardScaler, RobustScaler, BoxCoxTransformer

import scipy.stats as ss
from sklearn.pipeline import Pipeline


class SmartScaler(SmartTransformer):
    def _get_transformer(self, X, y=None, **fit_params):
        data = X.iloc[:, 0]
        cutoff = 0.05
        # statistically invalid but good enough measure of relative closeness
        # ks-test does not allow estimated parameters
        distributions = {"norm": StandardScaler(), "uniform": MinMaxScaler()}
        p_vals = {}
        for d in distributions.keys():
            dist = getattr(ss.distributions, d)
            p_vals[d] = ss.kstest(data, d, args=dist.fit(data)).pvalue
        best_dist = max(p_vals, key=p_vals.get)
        best_dist = best_dist if p_vals[best_dist] >= 0.05 else None
        if best_dist is None:
            return Pipeline(
                [("box_cox", BoxCoxTransformer()), ("robust_scaler", RobustScaler())]
            )
        else:
            return distributions[best_dist]
