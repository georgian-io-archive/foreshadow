"""Custom foreshadow defined transformers."""
from foreshadow.concrete.internals.cleaners import *  # noqa: F403, F401
from foreshadow.concrete.internals.cleaners import __all__ as c_all
from foreshadow.concrete.internals.dropfeature import DropFeature  # noqa: F401
from foreshadow.concrete.internals.dummyencoder import (  # noqa: F403, F401
    DummyEncoder,
)
from foreshadow.concrete.internals.fancyimpute import (  # noqa: F403, F401
    FancyImputer,
)
from foreshadow.concrete.internals.financial import (  # noqa: F401
    ConvertFinancial,
    PrepareFinancial,
)
from foreshadow.concrete.internals.htmlremover import HTMLRemover  # noqa: F401
from foreshadow.concrete.internals.labelencoder import (  # noqa: F403, F401
    FixedLabelEncoder,
)
from foreshadow.concrete.internals.nan_filler import NaNFiller  # noqa: F401
from foreshadow.concrete.internals.notransform import NoTransform  # noqa: F401
from foreshadow.concrete.internals.tfidf import (  # noqa: F403, F401
    FixedTfidfVectorizer,
)
from foreshadow.concrete.internals.tostring import ToString  # noqa: F401
from foreshadow.concrete.internals.uncommonremover import (  # noqa: F403, F401
    UncommonRemover,
)


# TODO flake fails here, figure out why.
#  hypothesis: flake8 uses the __repr__ which is modified to be
#  DFTransformer.HTMLRemover etc.

__all__ = [
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
    "NaNFiller",
] + c_all
