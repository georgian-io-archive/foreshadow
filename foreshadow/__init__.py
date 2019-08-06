"""An AutoML package to streamline the data science work flow."""

from foreshadow import console
from foreshadow.foreshadow import Foreshadow
from foreshadow.preparer import DataPreparer


__doc__ = """
foreshadow - Peer into the future of a data science project
===========================================================

TODO
"""


def get_version():
    import os
    import toml

    init_path = os.path.abspath(os.path.dirname(__file__))
    pyproject_path = os.path.join(init_path, "../pyproject.toml")

    with open(pyproject_path, "r") as fopen:
        pyproject = toml.load(fopen)

    return pyproject["tool"]["poetry"]["version"]


__version__ = get_version()
__all__ = ["Foreshadow", "DataPreparer", "console", "__version__"]
del get_version
