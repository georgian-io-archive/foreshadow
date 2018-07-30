from .transformers import SmartTransformer
from .transformers import Imputer
from sklearn.pipeline import Pipeline


class SimpleImputer(SmartTransformer):
    def __init__(self, threshold=0.1, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def _get_transformer(self, X, y=None, **fit_params):

        s = X.ix[:, 0]
        ratio = s.isnull().count() / s.count()

        if ratio <= self.threshold:
            return _choose_simple(s)
        else:
            return Pipeline([("null", None)])


class MultiImputer(SmartTransformer):
    def _get_transformer(self, X, y=None, **fit_params):
        return Imputer()


def _choose_simple(df):
    return Imputer()
