"""Classes for optimizing Foreshadow given a param_distribution."""

import six
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils import check_random_state

import foreshadow as fs
import foreshadow.serializers as ser
import hyperopt.pyll.stochastic as stoch
from foreshadow.base import BaseEstimator, TransformerMixin
from hyperopt import hp


"""
combinations:
    X_preparer.cleaner.CHAS:
        Cleaner:
            - date:
                - p1
                - p2
            - financial
        IntentMapper:
            - Something

    X_preparer.cleaner.CHAS.CleanerMapper:
        -Something

    X_preparer.cleaner.CHAS.IntentMapper:
        -Something


    X_preparer:
        cleaner:
            CHAS:
                Cleaner:
                    date:
                        -p1
                        -p2
                        
                        
Convention:
    Column name is last. If a .<blank> is present, then applied across all 
    columns.

Things that may be swapped:
    PreparerSteps,
    StepSmartTransformers/ConcreteTransformers.

"""


def _replace_list(key, obj, replace_with=hp.choice):
    """Recursively replace a nested object's lists with a sampling function.

    Replaces lists/tuples with replace_with.

    Args:
        key: Current key. Derived from dict keys in nested calls, but should
            be passed if your top level is a list.
        obj: the object to have list/tuples replaced.
        replace_with: Function that takes a key and list and builds a
            sampling function with it. Built around hp.choice but should be
            extendable.

    Returns:
        obj with lists/tuples replaced with replace_with.

    """
    key = str(key)
    if isinstance(obj, (tuple, list)):
        if not isinstance(obj[0], dict):
            #  we have reached a leaf of parameter specifications.
            return replace_with(key, obj)
        else:  # not a leaf, recurse and replace the output.
            to_replace = []
            for v in obj:
                to_replace.append(_replace_list(key, v, replace_with))
            return replace_with(key, to_replace)
    if isinstance(obj, dict):  # not a leaf for sure, we iterate over dict.
        to_replace = {}
        for key, v in obj.items():
            to_replace[key] = _replace_list(key, v, replace_with)
        obj.update(to_replace)
        return obj
    else:  # no nesting and no need to replace.
        return obj


class Tuner(BaseEstimator, TransformerMixin)
    """Tunes the forshadow object using a ParamSpec and Optimizer."""
    def __init__
