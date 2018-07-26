from sklearn.base import BaseEstimator, TransformerMixin
from .transformers import _get_classes, wrap_transformer
from .transformers import SmartTransformer
from .transformers import ParallelProcessor


def _get_modules():
    """Imports sklearn transformers from transformers directory.

    Searches transformers directory for classes implementing BaseEstimator and
    TransformerMixin and duplicates them, wraps their init methods and public functions
    to support pandas dataframes, and exposes them as foreshadow.transformers.[name]

    Returns:
        The number of transformers wrapped.

    """

    transformers = [
        cls
        for cls in _get_classes()
        if issubclass(cls, TransformerMixin) and issubclass(cls, BaseEstimator)
    ]

    for t in transformers:
        copied_t = type(t.__name__, t.__bases__, dict(t.__dict__))
        copied_t.__module__ = "foreshadow.transformers"
        globals()[copied_t.__name__] = wrap_transformer(copied_t)

    return len(transformers)


print("Loading transformers....")
n = _get_modules()
print("Loaded {} transformer plugins".format(n))
