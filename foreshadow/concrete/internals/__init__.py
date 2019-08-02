"""Custom foreshadow defined transformers."""

from foreshadow.concrete.internals.cleaners import *  # noqa: F403, F401
from foreshadow.concrete.internals.cleaners import __all__ as c_all
from foreshadow.concrete.internals.intents import *  # noqa: F401, F403
from foreshadow.concrete.internals.intents import __all__ as i_all
from foreshadow.concrete.internals.notransform import NoTransform  # noqa: F401

from .boxcox import BoxCox  # noqa: F401
from .dropfeature import DropFeature  # noqa: F401
from .dummyencoder import DummyEncoder  # noqa: F401
from .fancyimpute import FancyImputer  # noqa: F401
from .financial import ConvertFinancial, PrepareFinancial  # noqa: F401
from .htmlremover import HTMLRemover  # noqa: F401
from .labelencoder import FixedLabelEncoder  # noqa: F401
from .tfidf import FixedTfidfVectorizer  # noqa: F401
from .tostring import ToString  # noqa: F401
from .uncommonremover import UncommonRemover  # noqa: F401


__all__ = (
    [
        "BoxCox",
        "DropFeature",
        "DummyEncoder",
        "FancyImputer",
        "ConvertFinancial",
        "PrepareFinancial",
        "HTMLRemover",
        "FixedLabelEncoder",
        "FixedTfidfVectorizer",
        "ToString",
        "UncommonRemover",
        "YYYYMMDDDateCleaner",
        "DollarFinancialCleaner",
        "DropCleaner",
        "StandardJsonFlattener",
        "NoTransform",
        "Categoric",
        "Text",
        "Numeric",
        "BaseIntent",
    ]
    + c_all
    + i_all
)
