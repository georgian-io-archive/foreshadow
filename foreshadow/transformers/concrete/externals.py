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

from foreshadow.transformers.core import _get_modules


no_serialize_params = {"OneHotEncoder": ["cols"], "HashingEncoder": ["cols"]}


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
