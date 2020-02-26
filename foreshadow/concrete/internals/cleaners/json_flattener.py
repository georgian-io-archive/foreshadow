"""Detects and flattens json entries into various output styles."""
import json

import numpy as np
import pandas as pd

from .base import BaseCleaner


# from collections import MutableMapping


# TODO Comment out since we are not using it for now but may need it for future
#  dev.
# def flatten(d, parent_key="", sep="_"):
#     """Flatten the json completely, preserving tree in names.
#
#     Args:
#         d: dict to flatten.
#         parent_key: what to put for the at the beginning of the flattened key
#             Empty maps to the parent's value.
#         sep: Separate between 'parent_key' and the current key.
#
#     Returns:
#         dict with only 1 layer.
#
#     """
#     items = []
#     for k, v in d.items():
#         new_key = "{0}{1}{2}".format(parent_key, sep, k) if parent_key else k
#         if isinstance(v, MutableMapping):
#             items.extend(flatten(v, new_key, sep=sep).items())
#         elif isinstance(v, list):
#             # apply itself to each element of the list - that's it!
#             items.append((new_key, map(flatten, v)))
#         else:
#             items.append((new_key, v))
#     return dict(items)


def json_flatten(text):
    """Flatten a json array.

    Args:
        text: string of text

    Returns:
        length of match, new string assuming a match.
        Otherwise: None, original text.

    """
    ret = text
    matched = 0
    try:
        ret = json.loads(text)
        matched = len(text)
        ret = np.nan if _is_json_empty(ret) else ret
    except (json.JSONDecodeError, TypeError):
        pass  # didn't match.
    return ret, matched


def _is_json_empty(data):
    """Check if the json is an empty list or dictionary.

    Args:
        data: the json file to be checked.

    Returns:
        boolean -- whether the json file is empty.

    """
    if data == [] or data == {}:
        return True
    return False


class StandardJsonFlattener(BaseCleaner):
    """Clean financial data.

    Note: requires pandas input dataframes.

    """

    def __init__(self):
        transformations = [json_flatten]
        super().__init__(transformations)

    def metric_score(self, X: pd.DataFrame) -> float:
        """Compute the score for this cleaner using confidence_computation.

        confidence_computation is passed through init for each subclass.
        The confidence determines which cleaner/flattener is picked in an
        OVR fashion.

        Args:
            X: input DataFrame.

        Returns:
            float: confidence value.

        """
        score = super().metric_score(X)
        if score < 1:
            # we want to make sure the whole column is valid JSON. Otherwise
            # it will fail later steps. The reason we are not fixing the
            # JSON is because the variety of malformed JSON is unbounded.
            return 0
        return score
