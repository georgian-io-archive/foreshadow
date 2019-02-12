from .foreshadow import Foreshadow
from .preprocessor import Preprocessor

__doc__ = """
foreshadow - Peer into the future of a data science project
===========================================================

TODO
"""


def get_version():
    import toml

    with open("./pyproject.toml", "r") as fopen:
        pyproject = toml.load(fopen)

    return pyproject["tool"]["poetry"]["version"]


__version__ = get_version()

__all__ = ["Foreshadow", "Preprocessor", "__version__"]
