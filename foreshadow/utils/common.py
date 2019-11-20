"""Common utility functions."""

import os
from collections import OrderedDict
from importlib import import_module

from foreshadow.exceptions import TransformerNotFound
from foreshadow.utils.override_substitute import Override


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
                "foreshadow.cachemanager",
                "foreshadow.pipeline",
                "foreshadow.preparer",
                "foreshadow.estimators",
                "foreshadow.utils",
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


class ConfigureCacheManagerMixin:
    """Mixin that configure cache_manager."""

    def configure_cache_manager(self, cache_manager):
        """Configure the cache_manager attribute if exists.

        Args:
            cache_manager:  a cache_manager instance

        """
        if hasattr(self, "cache_manager"):
            self.cache_manager = cache_manager


class UserOverrideMixin:
    """Mixin that handles applying user override through force reresolve."""

    def should_force_reresolve_based_on_override(self, X):
        """Check if it should force reresolve based on user override.

        Args:
            X: the data frame

        Returns:
            bool: whether we should force reresolve based on user override.

        """
        if self._has_fitted() and self.cache_manager.has_override():
            """
            Note: If it is fitted and we have an intent override and we are
            dealing with a single column, then we do the following but if
            this is a group of columns, we may need to check if there are
            any columns in the override belong to this group, which is
            defined by X.columns.
            """
            if len(X.columns) == 1:
                override_key = "_".join([Override.INTENT, X.columns[0]])
                if override_key in self.cache_manager["override"]:
                    return True
            else:
                """need to iterate over the override dict. Not super
                efficient but not bad either as we don't expect too many
                overrides
                """
                for key in self.cache_manager["override"]:
                    if (
                        key.startswith(Override.INTENT)
                        and key.split("_")[1] in X.columns
                    ):
                        return True

        return False
