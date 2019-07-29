"""Base package for all intent definitions."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from foreshadow.metrics.metrics import metric

from foreshadow.utils import check_series

from pandas.api.types import is_numeric_dtype, is_string_dtype

from functools import partial


@metric(0)
def num_valid(X):
    X = check_series(X)
    data = ~X.apply(pd.to_numeric, errors="coerce").isnull()

    return float(data.sum()) / data.size

@metric(0)
def unique_heur(X):
    X = check_series(X)
    return 1 - (1.0 * X.nunique() / X.count())


@metric(0)
def is_numeric(X):
    X = check_series(X)
    return is_numeric_dtype(X)

@metric(0)
def is_string(X):
    X = check_series(X)
    return is_string_dtype(X)

class BaseIntent(BaseEstimator, TransformerMixin):
    @classmethod
    def get_confidence(cls, X, y=None):
        """Determine the confidence for an intent match.

        Args:
            X: input DataFrame.

        Returns:
            float: A confidence value bounded between 0.0 and 1.0

        """
        return sum(
            [
                metric_fn(X) * weight
                for metric_fn, weight in cls.confidence_computation.items()
            ]
        )


class Numeric(BaseIntent):
    confidence_computation = {
        num_valid: 0.25,
        partial(unique_heur, invert=True): 0.25,
        is_numeric: 0.25,
        partial(is_string, invert=True): 0.25
    }

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        return X.apply(pd.to_numeric, errors="coerce")


class Categoric(BaseIntent):
    confidence_computation = {
        num_valid: (1/3),
        unique_heur: (1/3),
        lambda x: 1: (1/3),
    }

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        return X


class Text(BaseIntent):
    confidence_computation = {
        num_valid: 0.25,
        unique_heur: 0.25,
        partial(is_numeric, invert=True): 0.25,
        is_string: 0.25
    }


    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        return X.astype(str)


if __name__ == '__main__':
    import numpy as np
    import pandas as pd

    num = pd.DataFrame(np.arange(100))
    cat = pd.DataFrame([1, 2, 3, 4, 5]*3)
    text = pd.DataFrame(['test', 'hello', 'I', 'python'])

    # print('Numeric Data:')
    # print('Numeric: {0:>12.3f}'.format(Numeric.get_confidence(num)))
    # print('Categoric: {0:>10.3f}'.format(Categoric.get_confidence(num)))
    # print('Text: {0:>15.3f}'.format(Text.get_confidence(num)))
    # 
    # print('Categoric Data:')
    # print('Numeric: {0:>12.3f}'.format(Numeric.get_confidence(cat)))
    # print('Categoric: {0:>10.3f}'.format(Categoric.get_confidence(cat)))
    # print('Text: {0:>15.3f}'.format(Text.get_confidence(cat)))
    
    # print('Text Data:')
    # print('Numeric: {0:>12.3f}'.format(Numeric.get_confidence(text)))
    # print('Categoric: {0:>10.3f}'.format(Categoric.get_confidence(text)))
    # print('Text: {0:>15.3f}'.format(Text.get_confidence(text)))