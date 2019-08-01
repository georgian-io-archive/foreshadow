"""External transformers.

All sklearn transformers imported here will be wrapped and made available in
the module :mod:`foreshadow.transformers.concrete`

"""

from category_encoders import HashingEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import (
    Imputer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from foreshadow.core import make_pandas_transformer
from foreshadow.utils import is_transformer


no_serialize_params = {"OneHotEncoder": ["cols"], "HashingEncoder": ["cols"]}


def _get_modules(classes, globals_, mname):  # TODO auto import all
    # TODO sklearn transformers and test each one generically.
    """Import sklearn transformers from transformers directory.

    Searches transformers directory for classes implementing BaseEstimator and
    TransformerMixin and duplicates them, wraps their init methods and public
    functions to support pandas dataframes, and exposes them as
    foreshadow.transformers.[name]

    Args:
        classes: A list of classes
        globals_: The globals in the callee's context
        mname: The module name
        wrap: True to wrap the transformers.

    Returns:
        The list of wrapped transformers.

    """
    # noqa: D202
    def no_wrap(t):
        """Return original function pointer.

        Don't wrap the transformer.

        Args:
            t: input transformer

        Returns:
            t, unwrapped.

        """
        return t

    transformers = [
        cls for cls in classes if is_transformer(cls, method="issubclass")
    ]

    for t in transformers:
        copied_t = type(t.__name__, (t, *t.__bases__), dict(t.__dict__))
        copied_t.__module__ = mname
        globals_[copied_t.__name__] = make_pandas_transformer(copied_t)

    return [t.__name__ for t in transformers]


def _get_classes():
    """Return a list of classes found in transforms directory.

    Returns:
        list of classes found in transforms directory.

    """
    import inspect

    return [c for c in globals().values() if inspect.isclass(c)]


classes = _get_modules(_get_classes(), globals(), __name__)
_all__ = [
    "HashingEncoder",
    "OneHotEncoder",
    "PCA",
    "TfidfVectorizer",
    "TfidfTransformer",
    "Imputer",
    "MinMaxScaler",
    "RobustScaler",
    "StandardScaler",
    "no_serialize_params",
]

del _get_classes
del _get_modules
