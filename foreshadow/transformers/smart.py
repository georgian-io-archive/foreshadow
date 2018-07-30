"""Smart Transformers

Transformers here will be accessible through the namespace
foreshadow.transformers.smart OR simple foreshadow.transformers and will not be
wrapped or transformed. Only classes extending SmartTransformer should exist here.

"""

from .imputer import SimpleImputer, MultiImputer
from .custom import BoxCoxTransformer

from ..transformers import SmartTransformer
from ..transformers import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler
)

import scipy.stats as ss

class SmartScaler(SmartTransformer):
    def _get_transformer(self, X, y=None, **fit_params):
        data = X.iloc[:, 0]
        cutoff = 0.05
        is_uniform = ss.kstest(data, 'uniform', args=ss.uniform.fit(data)).pvalue < cutoff
        is_normal = ss.kstest(data, 'norm', args=ss.norm.fit(data)).pvalue < cutoff

        # prefer normal, then uniform, or transform
        if is_normal:
            return StandardScaler()
        elif is_uniform:
            return MinMaxScaler()
        else:
            return Pipeline([('box_cox', BoxCoxTransformer()),
                             ('robust_scaler', RobustScaler())])
        