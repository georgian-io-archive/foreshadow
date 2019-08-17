"""Foreshadow optimizers."""

# from foreshadow.optimizers.param_mapping import param_mapping
from foreshadow.optimizers.param_distribution import ParamSpec
from foreshadow.optimizers.tuner import Tuner
from foreshadow.optimizers.random_search import RandomSearchCV
from foreshadow.utils import get_transformer

test_params = [
            {
                "s__transformer": "StandardScaler",
                "s__transformer__with_mean": [False,True],
            },
            {
                "s__transformer": "MinMaxScaler",
                "s__transformer__feature_range": [(0, 1), (0, 0.5)],
            },
        ]

test_params = [
            {
                "X_preparer__feature_preprocessor___parallel_process__group"
                ": 0__CategoricalEncoder__transformer__ohe":
                    get_transformer("OneHotEncoder")(),
                "X_preparer__feature_preprocessor___parallel_process__group"
                ": 0__CategoricalEncoder__transformer__ohe__drop_invariant":
                    [True, False],
            },
            {
                "X_preparer__feature_preprocessor___parallel_process__group"
                ": 0__CategoricalEncoder__transformer__ohe":
                    get_transformer("HashingEncoder")()
            },
        ]

__all__ = ["ParamSpec", "Tuner", "RandomSearchCV", "test_params"]

