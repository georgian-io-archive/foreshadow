"""Configuration file for AutoEstimators."""

from tpot.config.classifier import classifier_config_dict
from tpot.config.regressor import regressor_config_dict


def get_tpot_config(type_, include_preprocessors=False):
    """Get default configurations from TPOT.

    Drops feature engineering steps by default.

    Args:
        include_preprocessors (bool, optional): whether or not to include
            feature engineering steps.
        type_: type of classifier

    Returns:
        default config from TPOT

    Raises:
        ValueError: type_ not a valid type_

    """
    # TODO: allow TPOT to use xgboost again
    classifier_config_dict.pop("xgboost.XGBClassifier", None)
    regressor_config_dict.pop("xgboost.XGBRegressor", None)

    configs = {
        "classification": classifier_config_dict,
        "regression": regressor_config_dict,
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
        raise ValueError(
            "type_: '{0}' not in : '{1}'".format(type_, configs.keys())
        )
    return (
        {
            k: v
            for k, v in configs[type_].items()
            if not any(p in k for p in drop_partials)
        }
        if not include_preprocessors
        else configs[type_]
    )
