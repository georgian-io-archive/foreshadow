import json
from collections import MutableMapping

from foreshadow.cleaners.data_cleaner import BaseCleaner


def flatten(d, parent_key="", sep="_"):
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


def json_flatten(text, return_search=False):
    """Flatten a json array.

    Args:
        text: string of text

    Returns:
        length of match, new string assuming a match.
        Otherwise: None, original text.

    """
    #   regex = """
    # /
    # (?(DEFINE)
    #    (?<number>   -? (?= [1-9]|0(?!\d) ) \d+ (\.\d+)? ([eE] [+-]? \d+)? )
    #    (?<boolean>   true | false | null )
    #    (?<string>    " ([^"\\\\]* | \\\\ ["\\\\bfnrt\/] | \\\\ u [0-9a-f]{4} )* " )
    #    (?<array>     \[  (?:  (?&json)  (?: , (?&json)  )*  )?  \s* \] )
    #    (?<pair>      \s* (?&string) \s* : (?&json)  )
    #    (?<object>    \{  (?:  (?&pair)  (?: , (?&pair)  )*  )?  \s* \} )
    #    (?<json>   \s* (?: (?&number) | (?&boolean) | (?&string) | (?&array) | (?&object) ) \s* )
    # )
    # \A (?&json) \Z
    # /six
    #   """
    ret = text
    matched = 0
    try:
        ret = json.loads(text)
        matched = len(text)
    except:
        pass
    if return_search:
        return ret, matched
    return text


class StandardJsonFlattener(BaseCleaner):
    """Clean financial data.

    Note: requires pandas input dataframes.

    """

    def __init__(self):
        transformations = [json_flatten]
        super().__init__(transformations)
