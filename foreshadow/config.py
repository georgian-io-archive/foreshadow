"""Foreshadow system config resolver."""

import os

import yaml
import json

from foreshadow.utils import get_config_path, get_transformer

from collections import MutableMapping


CONFIG_FILE_NAME = "config.yml"

# TODO write a singleton object for this dictionary

DEFAULT_CONFIG = {
    "Cleaner": {
        'Flatteners': ['StandardJsonFlattener'],
        'Cleaners': ['YYYYMMDDDateCleaner', 'DropCleaner', 'DollarFinancialCleaner']
    },
    "Tiebreak": ["Numeric", "Categoric", "Text"],
    "Numeric": {"Preprocessor": ["Imputer", "Scaler"]},
    "Categoric": {"Preprocessor": ["CategoricalEncoder"]},
    "Text": {"Preprocessor": ["TextEncoder"]},
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
    """Defines a single-instance config store with convenience methods."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.system_config = DEFAULT_CONFIG
        self.user_config = load_config(get_config_path())
        self._cfg_list = {} # key is path

    def get_config(self):
        """Resolves config at init time."""
        local_path = os.path.abspath("")
        local_config = load_config(local_path)

        # global _cfg
        # if local_path in _cfg:
        #     return _cfg.get(local_path)

        # Expand the dictionaries in order of precedence
        resolved_strs = {
            **self.system_config,
            **self.user_config,
            **local_config
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

    def get_intents(self):
        return self.get_config()['Tiebreak']

    def get_preprocessor_steps(self, intent):
        return self.get_config()[intent]['Preprocessor']

    def clear(self):
        self._cfg_list = {}

    def __delitem__(self, key):
        del self._cfg_list[key]

    def __getitem__(self, key):
        return self._cfg_list[key]

    def __iter__(self):
        for data in self._cfg_list:
            yield data

    def __len__(self):
        return len(self._cfg_list)

    def __setitem__(self):
        raise NotImplementedError("The config cannot be manually set.")

    def __eq__(self, other):
        return self._cfg_list == other

config = ConfigStore()
