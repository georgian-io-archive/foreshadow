"""DropCleaner which detects when to drop cleaner."""

import pandas as pd

from foreshadow.metrics import (
    MetricWrapper,
    calculate_percentage_of_rows_matching_regex,
)
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
    if text is None or text == "" or pd.isna(text):
        # Need to check np.isnan last as it fails on empty string.
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
        super().__init__(
            transformations,
            confidence_computation={
                MetricWrapper(calculate_percentage_of_rows_matching_regex): 1
            },
        )

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
        if score < 0.9:
            # only drop a column if 90% of the data is NaN
            return 0
        return score

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
        return pd.DataFrame([], columns=X.columns, index=X.index)
