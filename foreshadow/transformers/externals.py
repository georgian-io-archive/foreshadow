"""External Transformers

All sklearn transformers imported here will be wrapped and made available in the
module foreshadow.transformers

"""

import inspect

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Imputer
from sklearn.decomposition import PCA
from category_encoders import HashingEncoder

from .transformers import _get_modules


def _get_classes():
    """Returns list of classes found in transforms directory."""
    return [c for c in globals().values() if inspect.isclass(c)]


classes = _get_modules(_get_classes(), globals(), __name__)
