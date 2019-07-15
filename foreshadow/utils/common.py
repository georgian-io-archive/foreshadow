"""Common utility functions."""

import os


DATA_DIR = "~/.foreshadow"


def get_config_path():
    """Get the default config path.

    Note:
        This function also makes the directory if it does not already exist.

    Returns:
        str: The path to the config directory.

    """
    ret_path = os.path.expanduser(DATA_DIR)
    os.makedirs(ret_path, exist_ok=True)

    return ret_path


def get_cache_path(path=None):
    """Get the cache path which is in the config directory.

    Note:
        This function also makes the directory if it does not already exist.

    Args:
        path (str): A path to override the cache save directory path.

    Returns:
        str; The path to the cache directory.

    """
    cache_path = os.path.join(get_config_path(), "cache")
    os.makedirs(cache_path, exist_ok=True)

    return cache_path
