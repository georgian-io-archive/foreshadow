"""
Configuration file for auto estimators
"""

import numpy as np

from tpot.config.classifier import classifier_config_dict as tpot_classifier_config
from tpot.config.regressor import regressor_config_dict as tpot_regressor_config


def get_tpot_config(type_, include_preprocessors=False):
    configs = {
        "classification": tpot_classifier_config,
        "regression": tpot_regressor_config,
    }

    drop_partials = [
        "preprocessing",
        "kernel_approximation",
        "decomposition",
        "builtins",
        "feature_selection",
        "cluster",
    ]
    if type_ not in configs.keys():
        raise ValueError("type_ must be either classification or regression")
    return (
        {
            k: v
            for k, v in configs[type_].items()
            if not any(p in k for p in drop_partials)
        }
        if not include_preprocessors
        else configs[type_]
    )
