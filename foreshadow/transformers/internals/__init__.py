"""Custom foreshadow defined transformers."""

import glob
import inspect
import os

from foreshadow.transformers.transformers import _get_modules


def _get_classes():
    """Return list of classes found in transforms directory.

    Returns:
        list of classes found in transforms directory

    """
    files = glob.glob(os.path.dirname(__file__) + "/*.py")
    imports = [
        os.path.basename(f)[:-3]
        for f in files
        if os.path.isfile(f) and not f.endswith("__init__.py")
    ]
    modules = [
        __import__(i, globals(), locals(), ["object"], 1) for i in imports
    ]
    classes = [
        c[1]
        for m in modules
        for c in inspect.getmembers(m)
        if inspect.isclass(c[1])
    ]

    return classes


classes = _get_modules(_get_classes(), globals(), __name__)
__all__ = classes
