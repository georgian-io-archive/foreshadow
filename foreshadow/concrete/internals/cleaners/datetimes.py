"""Transforms for datetime inputs."""
import re

from .base import BaseCleaner


def _split_to_new_cols(t):
    """Clean text if it is in a YYYYMDD format and split to three columns.

    Args:
        t: string of text

    Returns:
        length of match, new string assuming a match.
        Otherwise: None, original text.

    """
    delimiters = "[-/]"
    regex = r"^.*(([\d]{{4}}){delim}([\d]{{2}}){delim}([\d]{{2}})).*$".format(
        delim=delimiters
    )
    text = str(t)
    res = re.search(regex, text)
    if res is not None:
        res = sum([len(range(reg[0], reg[1])) for reg in res.regs[1:2]])
        texts = [re.sub(regex, r"\{}".format(i), text) for i in range(2, 5)]
    else:
        texts = t
        res = 0
    return texts, res


# Due to Parallel processing issue, this method cannot be a local method.
def make_list_of_three(x):
    """Return default output which must have 3 columns.

    Args:
        x: initial row.

    Returns:
        list of three with initial row as first element, 2 empty
        elements.

    """
    return [x, "", ""]


class YYYYMMDDDateCleaner(BaseCleaner):
    """Clean DateTime data.

    Note: requires pandas input dataframes.

    """

    def __init__(self):
        transformations = [_split_to_new_cols]
        default = make_list_of_three
        super().__init__(transformations, default=default)
