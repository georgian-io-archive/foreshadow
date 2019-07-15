"""Utilities used when testing foreshadow."""

import os
import sys
from collections import OrderedDict
from functools import lru_cache
from importlib import import_module


@lru_cache(maxsize=None)
def _get_test_folder():
    """Get path of the main test folder.

    Path is assumed to be located somewhere above this file. This computation
    is cached as the absolute directory of the cache isn't expected to change.

    Returns:
        path to test folder (root for tests)

    Raises:
        FileNotFoundError: If tests folder could not be found.
        RuntimeError: If more than one tests folder is found at the same time.

    """
    path = os.path.abspath(os.path.dirname(__file__))
    while len(path) > 1:
        find_test_dir = [
            d
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d)) and d == "tests"
        ]
        if len(find_test_dir) == 1:
            return os.path.join(path, find_test_dir[0])
        elif len(find_test_dir) > 1:
            raise RuntimeError("Found more than one tests directory")
        else:
            path = os.path.dirname(path)
    raise FileNotFoundError("Could not find tests directory in path")


def get_file_path(file_type, file_name):
    """Get the path to a file inside of tests.

    Useful for paths to static files, such as data or configs.

    Args:
        file_type: Identifies where the file is stored.
        file_name: Name of file.

    Returns:
        path to file

    """
    test_path = _get_test_folder()
    return os.path.join(test_path, file_type, file_name)


def debug():  # noqa: D202
    """Add pdb debugger on import.

    Utility to add pdb debugging to an entire file so that on error, the pdb
    utility is opened.
    """

    def _info(type, value, tb):
        # Source: https://stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error # noqa
        if hasattr(sys, "ps1") or not sys.stderr.isatty():
            sys.__excepthook__(type, value, tb)
        else:
            import traceback
            import pdb

            traceback.print_exception(type, value, tb)
            pdb.post_mortem(tb)

    sys.excepthook = _info


def get_transformer(class_name, source_lib=None):
    """Get the transformer class from its name.

    Note:
        In case of name conflict, internal transformer is preferred over
        external transformer import. This should only be using in internal
        unit tests, get_transformer from serialization should be preferred in
        all other cases. This was written to decouple registration from unit
        testing.

    Args:
        class_name (str): The transformer class name
        source_lib (str): The string import path if known

    Returns:
        Imported class

    Raises:
        ValueError: If class_name could not be found in internal or external
            transformer library pathways.

    """
    if source_lib is not None:
        module = import_module(source_lib)
    else:
        sources = OrderedDict(
            (source, import_module(source))
            for source in [
                "foreshadow.transformers.concrete",
                "foreshadow.transformers.smart",
            ]
        )

        for v in sources.values():
            if hasattr(v, class_name):
                module = v
                break
        else:
            raise ValueError(
                "Could not find transformer {} in {}".format(
                    class_name, ", ".join(sources.keys())
                )
            )

    return getattr(module, class_name)
