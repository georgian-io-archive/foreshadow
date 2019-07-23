"""Foreshadow system config resolver."""

import os

import yaml

from foreshadow.utils import get_config_path, get_transformer


CONFIG_FILE_NAME = "config.yml"

DEFAULT_CONFIG = {
    "cleaner": [],
    "engineerer": {},
    "preprocessor": {
        "numerical": ["Imputer", "Scaler"],
        "categorical": ["CategoricalEncoder"],
        "text": ["TextEncoder"],
    },
    "reducer": {},
}


def get_config(base):
    """Get a configuration file given a root path.

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


def resolve_config():
    """Resolve the configuration to actual classes.

    Note:
        The order is resolution is as follows in increasing precedence order:
        framework, user, local.

    Returns:
        A dictionary with the same keys as `foreshadow.config.DEFAULT_CONFIG`
        with the correct overrides.

    """
    default = DEFAULT_CONFIG
    user = get_config(get_config_path())
    local = get_config(os.path.abspath(""))

    # Expand the dictionaries in order of precedence
    _resolved = {**default, **user, **local}

    resolved = {}
    for step, data in _resolved.items():
        if not len(data):
            resolved[step] = data
        elif isinstance(data, list):
            resolved[step] = [
                get_transformer(transformer) for transformer in data
            ]
        elif isinstance(data, dict):
            resolved[step] = {
                intent: [
                    get_transformer(transformer)
                    for transformer in transformer_list
                ]
                for intent, transformer_list in data.items()
            }

    return resolved
