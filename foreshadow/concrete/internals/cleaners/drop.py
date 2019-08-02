"""DropCleaner which detects when to drop cleaner."""
import re

import pandas as pd

from foreshadow.utils.validation import check_df

from .base import BaseCleaner


def drop_transform(text):
    """Drop this column at the cleaning stage.

    Args:
        text: string of text

    Returns:
        length of match, new string assuming a match.
        Otherwise: None, original text.

    """
    regex = "^$"
    text = str(text)
    res = re.search(regex, text)
    if res is not None:
        res = 1
    else:
        res = 0
    return text, res


class DropCleaner(BaseCleaner):
    """Clean financial data.

    Note: requires pandas input dataframes.

    """

    def __init__(self):
        transformations = [drop_transform]
        super().__init__(transformations)

    def transform(self, X, y=None):
        """Clean string columns.

        Here, we assume that any list output means that these are desired
        to be new columns in our dataset. Contractually, this could change
        to be that a boolean flag is passed to indicate when this is
        desired, as of right now, there should be no need to return a list
        for any case other than this case of desiring new column.

        The same is assumed for dicts, where the key is the new column name,
        the value is the value for that row in that column. NaNs
        are automatically put into the columns that don't exist for given rows.

        Args:
            X (:obj:`pandas.Series`): X data
            y: input labels

        Returns:
            :obj:`pandas.DataFrame`: Transformed data

        """
        X = check_df(X, single_column=True)
        return pd.DataFrame([], columns=X.columns)
