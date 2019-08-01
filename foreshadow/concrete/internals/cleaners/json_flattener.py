"""Detects and flattens json entries into various output styles."""
import json
from collections import MutableMapping

from .base import BaseCleaner


def flatten(d, parent_key="", sep="_"):
    """Flatten the json completely, preserving tree in names.

    Args:
        d: dict to flatten.
        parent_key: what to put for the at the beginning of the flattened key.
            Empty maps to the parent's value.
        sep: Separate between 'parent_key' and the current key.

    Returns:
        dict with only 1 layer.

    """
    items = []
    for k, v in d.items():
        new_key = "{0}{1}{2}".format(parent_key, sep, k) if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # apply itself to each element of the list - that's it!
            items.append((new_key, map(flatten, v)))
        else:
            items.append((new_key, v))
    return dict(items)


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
    except (json.JSONDecodeError, TypeError):
        pass  # didn't match.
    return ret, matched


class StandardJsonFlattener(BaseCleaner):
    """Clean financial data.

    Note: requires pandas input dataframes.

    """

    def __init__(self):
        transformations = [json_flatten]
        super().__init__(transformations)
