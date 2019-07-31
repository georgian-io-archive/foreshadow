"""Custom foreshadow defined transformers."""

from foreshadow.transformers.concrete.internals.boxcox import BoxCox
from foreshadow.transformers.concrete.internals.dropfeature import DropFeature
from foreshadow.transformers.concrete.internals.dummyencoder import (
    DummyEncoder,
)
from foreshadow.transformers.concrete.internals.fancyimpute import FancyImputer
from foreshadow.transformers.concrete.internals.financial import (
    ConvertFinancial,
    PrepareFinancial,
)
from foreshadow.transformers.concrete.internals.htmlremover import HTMLRemover
from foreshadow.transformers.concrete.internals.labelencoder import (
    FixedLabelEncoder,
)
from foreshadow.transformers.concrete.internals.tfidf import (
    FixedTfidfVectorizer,
)
from foreshadow.transformers.concrete.internals.tostring import ToString
from foreshadow.transformers.concrete.internals.uncommonremover import (
    UncommonRemover,
)


def _get_classes():
    """Return a list of classes found in transforms directory.

    Returns:
        list of classes found in transforms directory.

    """
    import inspect

    return [c for c in globals().values() if inspect.isclass(c)]


__all__ = [
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
    "UncommonRemover"
]

