"""StandardDollarFinancial transformers."""

import re

from .base import BaseCleaner


def financial_transform(text):
    """Clean text if it is a financial text.

    Args:
        text: string of text

    Returns:
        length of match, new string assuming a match.
        Otherwise: None, original text.

    """
    regex = r"^([\W]*\$)([\d]+[\.]?[\d]*)(.*)$"
    text = str(text)
    res = re.search(regex, text)
    if res is not None:
        res = sum([len(range(reg[0], reg[1])) for reg in res.regs[1:]])
        text = re.sub(regex, r"\2", text)
    else:
        res = 0
    return text, res


class DollarFinancialCleaner(BaseCleaner):
    """Clean financial data.

    Note: requires pandas input dataframes.

    """

    def __init__(self):
        transformations = [financial_transform]
        super().__init__(transformations)
