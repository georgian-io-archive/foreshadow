"""Utilities used when testing foreshadow."""

import os
import sys
from functools import lru_cache


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
        last_path = path
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
            if path == last_path:
                break
    raise FileNotFoundError("Could not find tests directory in path")


def get_file_path(data_folder, file_name):
    """Get the path to a file inside of tests.

    Useful for paths to static files, such as data or configs.

    Args:
        data_folder: Identifies where the file is stored.
        file_name: Name of file.

    Returns:
        path to file

    """
    test_path = _get_test_folder()
    return os.path.join(test_path, data_folder, file_name)


def debug():  # noqa: D202  # pragma: no cover
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


def dynamic_import(attribute, module_path):
    """Import attribute from module found at module_path at runtime.

    Args:
        attribute: the attribute of the module to import (class, function, ...)
        module_path: the path to the module.

    Returns:
        attribute from module_path.

    """
    from importlib import import_module

    mod = import_module(module_path)
    return getattr(mod, attribute)


def import_init_transformer(
    transformer_class,
    path="foreshadow.transformers.concrete",
    instantiate=True,
    params=None,
):
    """Import and init a transformer from a specified path.

    Args:
        transformer_class (str): The transformer class to import
        path (str): The import path to import from, default is
            `foreshadow.transformers.concrete`
        instantiate (bool): Whether or not to instantiate the class
        params (dict): A param dictionary
        instantiate:  TODO @Adithya

    Returns:
        object: an initialized version of the transformer

    """
    if instantiate:
        if params is not None:
            return dynamic_import(transformer_class, path)(**params)
        else:
            return dynamic_import(transformer_class, path)()
    else:
        return dynamic_import(transformer_class, path)
