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

_cfg = {}


def get_config(base):
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


def reset_config():
    """Reset internal configuration

    Note:
        This is useful in an IDLE setting when the configuration file might
        have been modified but you don't want to reload the system.

    """
    global _cfg
    _cfg = {}


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
    local_path = os.path.abspath("")
    local = get_config(local_path)

    # import pdb; pdb.set_trace()

    global _cfg
    if local_path in _cfg:
        return _cfg.get(local_path)

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

    _cfg[local_path] = resolved

    return resolved
