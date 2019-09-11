"""Common utility functions."""

import os
from collections import OrderedDict
from importlib import import_module

from foreshadow.exceptions import TransformerNotFound


CONFIG_DIR = "~/.foreshadow"


def get_config_path():
    """Get the default config path.

    Note:
        This function also makes the directory if it does not already exist.

    Returns:
        str: The path to the config directory.

    """
    ret_path = os.path.expanduser(CONFIG_DIR)
    os.makedirs(ret_path, exist_ok=True)

    return ret_path


def get_cache_path():
    """Get the cache path which is in the config directory.

    Note:
        This function also makes the directory if it does not already exist.

    Returns:
        str; The path to the cache directory.

    """
    cache_path = os.path.join(get_config_path(), "cache")
    os.makedirs(cache_path, exist_ok=True)

    return cache_path


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
        TransformerNotFound: If class_name could not be found in internal or
            external transformer library pathways.

    """
    if source_lib is not None:
        module = import_module(source_lib)
    else:
        sources = OrderedDict(
            (source, import_module(source))
            for source in [
                "foreshadow.concrete",
                "foreshadow.smart",
                "foreshadow.intents",
                "foreshadow.steps",
                "foreshadow.parallelprocessor",
                "foreshadow.columnsharer",
                "foreshadow.pipeline",
                "foreshadow.preparer",
                "foreshadow.estimators",
            ]
        )

        for v in sources.values():
            if hasattr(v, class_name):
                module = v
                break
        else:
            raise TransformerNotFound(
                "Could not find transformer {} in {}".format(
                    class_name, ", ".join(sources.keys())
                )
            )

    return getattr(module, class_name)


class ConfigureColumnSharerMixin:
    """Mixin that configure column sharer."""

    def configure_column_sharer(self, column_sharer):
        """Configure the column sharer attribute if exists.

        Args:
            column_sharer:  a column sharer instance

        """
        if hasattr(self, "column_sharer"):
            self.column_sharer = column_sharer


class CustomizeParamsMixin:
    """Mixin that customizes params returned from get_params."""

    def customize_serialization_params(self, params):
        """Select params that are only necessary.

        Args:
            params: the params returned from get_params

        Returns:
            selected_params: params that are selected.

        """
        import inspect

        init_params = inspect.signature(self.__init__).parameters
        selected_params = {
            name: params.pop(name)
            for name in init_params
            if name not in ["self", "kwargs"]
        }
        return selected_params
