import os


def _get_test_folder():
    """Get path of the main test folder.

    Path is assumed to be somewhere above this file.

    Returns:
        path to test folder (root for tests)

    Raises:
        FileNotFoundError: if tests folder could not be found

    """
    path = os.path.dirname(__file__)
    while len(path) > 0:
        folder = os.path.basename(path)
        if folder == "tests":
            return path
        else:
            path = os.path.dirname(path)
    raise FileNotFoundError("Could not find tests directory in path")


def get_file_path(file_type, file_name):
    """Gets the path to a file inside of tests.

    Useful for paths to static files, such as data or configs.

    Args:
        file_type: identifies where the file is stored
        file_name: name of file.

    Returns:
        path to file

    """
    test_path = _get_test_folder()
    return os.path.join(test_path, file_type, file_name)


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
    path="foreshadow.transformers.externals",
    instantiate=True,
    params=None,
):
    """Import and init a transformer from a specified path

    Args:
        transformer_class (str): The transformer class to import
        path (str): The import path to import from, default is
            `foreshadow.transformers.externals`
        params (dict): A param dictionary

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
