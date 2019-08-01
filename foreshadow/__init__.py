"""An AutoML package to streamline the data science work flow."""

# # Make sure to remove temporary F401
# from foreshadow.foreshadow import Foreshadow
# from foreshadow.preprocessor import Preprocessor
# from foreshadow import console

# This is temporary
from foreshadow.foreshadow import Foreshadow
from foreshadow.preprocessor import Preprocessor
from foreshadow import console


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

# __all__ = ["Foreshadow", "Preprocessor", "console", "__version__"]

__all__ = ["Foreshadow", "Preprocessor", "console", "__version__"]

del get_version
