"""Transforms for datetime inputs."""
import re

from foreshadow.cleaners.data_cleaner import BaseCleaner


def _split_to_new_cols(text, return_search=False):
    """Clean text if it is in a YYYYMDD format and split to three columns.

    Args:
        text: string of text

    Returns:
        length of match, new string assuming a match.
        Otherwise: None, original text.

    """
    delimiters = "[-/]"
    regex = "^.*(([\d]{4})%s([\d]{2})%s([\d]{2})).*$" % (
        delimiters,
        delimiters,
    )
    text = str(text)
    res = re.search(regex, text)
    if res is not None:
        res = sum([len(range(reg[0], reg[1])) for reg in res.regs[1:2]])
        texts = [re.sub(regex, r"\%d" % i, text) for i in range(2, 5)]
    if return_search:
        return texts, res
    return texts


class YYYYMMDDDateCleaner(BaseCleaner):
    """Clean financial data.

    Note: requires pandas input dataframes.

    """

    def __init__(self):
        transformations = [_split_to_new_cols()]
        super().__init__(transformations)
