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
from foreshadow.transformers.core.wrapper import _get_modules


def _get_classes():
    """Return a list of classes found in transforms directory.

    Returns:
        list of classes found in transforms directory.

    """
    import inspect

    return [c for c in globals().values() if inspect.isclass(c)]


classes = _get_modules(_get_classes(), globals(), __name__)
__all__ = classes

del _get_modules
