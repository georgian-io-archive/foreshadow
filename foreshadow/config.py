"""Foreshadow system config resolver."""

import json
import os
from collections import MutableMapping

import yaml

from foreshadow.utils import get_config_path, get_transformer


CONFIG_FILE_NAME = "config.yml"

_DEFAULT_CONFIG = {
    "Cleaner": {
        "Flatteners": ["StandardJsonFlattener"],
        "Cleaners": [
            "YYYYMMDDDateCleaner",
            "DropCleaner",
            "DollarFinancialCleaner",
        ],
    },
    "Tiebreak": ["Numeric", "Categorical", "Text"],
    "Numeric": {"Preprocessor": ["SimpleFillImputer", "Scaler"]},
    "Categorical": {"Preprocessor": ["CategoricalEncoder"]},
    "Text": {"Preprocessor": ["TextEncoder"]},
    "Droppable": {"Preprocessor": []},
}


def load_config(base):
    """Try to load configuration data from specific folder path.

    Args:
        base (str): A base path that has a file called `config.yml`

    Returns:
        dict: If the file does not exist an empty dictionary is returned.

    """
    data_file_path = os.path.join(base, CONFIG_FILE_NAME)
    check_file = os.path.exists(data_file_path) and os.path.isfile(
        data_file_path
    )

    if not check_file:
        return {}
    else:
        with open(data_file_path) as fopen:
            data = yaml.safe_load(fopen)
            if data is None:
                return {}
            else:
                return data


class ConfigStore(MutableMapping):
    """Define a single-instance config store with convenience methods.

    Attributues:
        system_config: The default system configuration dictionary
        user_config: The specific user configuration dictionary

    """

    def __init__(self, *args, **kwarg):
        self.system_config = _DEFAULT_CONFIG
        self.user_config = load_config(get_config_path())
        self._cfg_list = {}  # key is path
        self._multiprocess = False

    def get_n_jobs_config(self):
        """Determine the number of processes to use for ParallelProcessor.

        Returns:
            The number of processes to use by the ParallelProcessor.

        """
        return -1 if self._multiprocess else 1

    def set_multiprocess(self, status):
        """Configure whether to enable multiprocessing.

        Args:
            status: bool value.

        """
        self._multiprocess = status

    def get_config(self):
        """Resolve a config instance.

        Returns:
            dict: A resolved version of the system configuration that merges \
                system, user, and local configuration setups.

        """
        local_path = os.path.abspath("")
        local_config = load_config(local_path)

        # Expand the dictionaries in order of precedence
        resolved_strs = {
            **self.system_config,
            **self.user_config,
            **local_config,
        }

        resolved_hash = hash(json.dumps(resolved_strs, sort_keys=True))

        if resolved_hash in self._cfg_list:
            return self._cfg_list[resolved_hash]

        resolved = {}
        # key is cleaner, resolver, or intent
        # all individual steps are converted to classes
        for key, data in resolved_strs.items():
            if not len(data):
                resolved[key] = data
            elif isinstance(data, list):
                resolved[key] = [
                    get_transformer(transformer) for transformer in data
                ]
            elif isinstance(data, dict):
                resolved[key] = {
                    step: [
                        get_transformer(transformer)
                        for transformer in transformer_list
                    ]
                    for step, transformer_list in data.items()
                }

        self._cfg_list[resolved_hash] = resolved

        return resolved

    def get_cleaners(self, flatteners=False, cleaners=False):
        """Get cleaner setup.

        Args:
            flatteners (bool): get flatteners
            cleaners (bool): get cleaners

        Returns:
            list: A list of all the relavent classes

        Raises:
            ValueError: Both flatteners and cleaners cannot be false

        """
        if not (flatteners or cleaners):
            raise ValueError("Both flatteners and cleaners cannot be false.")

        config = self.get_config()

        flatteners = config["Cleaner"]["Flatteners"] if flatteners else []
        cleaners = config["Cleaner"]["Cleaners"] if cleaners else []

        return [*flatteners, *cleaners]

    def get_intents(self):
        """Get the intent resolution order.

        Returns:
            list: A list of intent objects in order

        """
        return self.get_config()["Tiebreak"]

    def get_preprocessor_steps(self, intent):
        """Get the preprocessor list for a given intent.

        Args:
            intent: A string of the intent to select upon.

        Returns:
            list: A list of transformation classes for an intent

        """
        return self.get_config()[intent]["Preprocessor"]

    def clear(self):
        """Clear all cached configuration stores."""
        self._cfg_list = {}

    def __delitem__(self, key):
        """Delete an item from the config cache for a given hash value.

        Args:
            key: A hash value for the item to delete

        """
        del self._cfg_list[key]

    def __getitem__(self, key):
        """Get an item from the config cache for a given hash value.

        Args:
            key: A hash value for the item to get

        Returns:
            dict: The configuration for a particular hash value.

        """
        return self._cfg_list[key]

    def __iter__(self):
        """Get the iterable the config cache.

        Yields:
            The key, value pairs of hash and its associated configuration.

        """
        for data in self._cfg_list:
            yield data

    def __len__(self):
        """Get the number of hashes saved in the cache.

        Returns:
            The number of hashes saved in the internal cache.

        """
        return len(self._cfg_list)

    def __setitem__(self):
        """Values cannot be set in the cache.

        Raises:
            NotImplementedError: The config cannot be manually set.

        """
        raise NotImplementedError("The config cannot be manually set.")

    def __eq__(self, other):
        """Check the equality of the cache with another cache instance.

        Args:
            other: Another cache instance or dictionary.

        Returns:
            bool: True if the caches are equal, False otherwise

        """
        return self._cfg_list == other


config = ConfigStore()
