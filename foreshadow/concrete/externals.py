"""External transformers.

All sklearn transformers imported here will be wrapped and made available in
the module :mod:`foreshadow.transformers.concrete`

"""

from category_encoders import HashingEncoder, OneHotEncoder  # noqa: F401
from sklearn.decomposition import PCA  # noqa: F401
from sklearn.feature_extraction.text import (  # noqa: F401
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.impute import SimpleImputer  # noqa: F401
from sklearn.preprocessing import (  # noqa: F401
    LabelEncoder,
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from foreshadow.utils import is_transformer
from foreshadow.wrapper import pandas_wrap


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

    Returns:
        The list of wrapped transformers.

    """
    transformers = [
        cls
        for cls in classes
        if is_transformer(cls, method="issubclass")  # noqa: F821
    ]  # flake does not detect due to del.

    for t in transformers:
        copied_t = type(t.__name__, (t, *t.__bases__), dict(t.__dict__))
        copied_t.__module__ = mname
        globals_[copied_t.__name__] = pandas_wrap(  # noqa: F821
            copied_t  # noqa: F821
        )
        # flake does not detect due to del.

    return [t.__name__ for t in transformers]


def _get_classes():
    """Return a list of classes found in transforms directory.

    Returns:
        list of classes found in transforms directory.

    """
    import inspect

    return [c for c in globals().values() if inspect.isclass(c)]


__all__ = _get_modules(_get_classes(), globals(), __name__) + [
    "no_serialize_params"
]

del pandas_wrap
del is_transformer
del _get_classes
del _get_modules
