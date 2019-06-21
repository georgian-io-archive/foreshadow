"""An end-to-end AutoML package to streamline the datascience workflow."""

from foreshadow.foreshadow import Foreshadow
from foreshadow.preprocessor import Preprocessor


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

__all__ = ["Foreshadow", "Preprocessor", "console", "__version__"]
