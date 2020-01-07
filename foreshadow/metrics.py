"""Metrics used across Foreshadow for smart decision making."""

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

from foreshadow.logging import logging
from foreshadow.utils import check_series


class MetricWrapper:
    """Class that wraps around the metric calculation function.

    Add default_return value as well as the option to calculate the
    inverted metric.

    """

    def __init__(self, fn, default_return=0, invert=False):
        self.fn = fn
        self.default_return = default_return
        self.invert = invert
        self._last_call = None

    def calculate(self, feature, **kwargs):
        """Use the metric function passed at initialization.

        Note:
            If default_return was set, the wrapper will suppress any errors
            raised by the wrapped function.

        Args:
            feature: feature/column of pandas dataset requires it.
            **kwargs: any keyword arguments to metric function

        Returns:
            The metric computation defined by the metric.

        Raises:
            re_raise: If default return is not set the metric then re-raise

        """
        try:
            self._last_call = self.fn(feature, **kwargs)
        except Exception as re_raise:
            logging.debug(
                "There was an exception when calling {}".format(self.fn)
            )
            if self.default_return is not None:
                return self.default_return
            else:
                raise re_raise

        return self._last_call if not self.invert else (1.0 - self._last_call)

    def last_call(self):
        """Value from previous call to metric function.

        Returns:
            Last call to metric_fn (self.fn)

        """
        return self._last_call

    def __str__(self):
        """Pretty print.

        Returns:
            $class.$fn

        """
        return "{0}.{1}".format(self.__class__.__name__, self.fn.__name__)

    def __repr__(self):
        """Unambiguous print.

        Returns:
            <$class, $fn, $id>

        """
        return "{0} with function '{1}' object at {2}>".format(
            str(self.__class__)[:-1], self.fn.__name__, id(self)
        )


def unique_count(feature):
    """Count number of unique values in feature.

    Args:
        feature: feature/column of pandas dataset

    Returns:
        number of unique values in feature as int.

    """
    return len(feature.value_counts())


def unique_count_bias(feature):
    """Difference of count of unique values relative to the length of feature.

    Args:
        feature: feature/column of pandas dataset

    Returns:
        Number of unique values relative to the length of dataset.

    """
    return len(feature) - len(feature.value_counts())


def unique_count_weight(feature):
    """Normalize count number of unique values relative to length of feature.

    Args:
        feature: feature/column of pandas dataset

    Returns:
        Normalized Number of unique values relative to length of feature.

    """
    return len(feature.value_counts()) / len(feature)


def calculate_percentage_of_rows_matching_regex(feature, cleaner):
    """Return percentage of rows matched by regex transformations.

    Cleaner(row) will return a CleanerReturn namedtupled, which will have the
    transformed text after all transformations (or the original text if it
    failed) and then a list of all the number of characters matched by each
    individual regex transform. Ergo, if any one of them is 0, then the
    transformations failed.

    Args:
        feature: a column of the dataset
        cleaner: callable that will perform all transformations to row of
            feature

    Returns:
        Return percentage of rows matched by regex transformations.

    """
    f = feature
    matched_lens = [cleaner(f.at[i, f.columns[0]]).match_lens for i in f.index]
    return sum([min(list_lens) for list_lens in matched_lens]) / len(feature)


def avg_col_regex(feature, cleaner, mode=min):
    """Return average percentage of each row's text transformed by cleaner.

    Cleaner will be a list of transformations, that will take the original
    text and transform it to new text. The percentage of the original text
    that is kept is averaged across all rows using 'mode'. For instance,
    if the original text was 'hello' and transformation 1 mapped 'll' to 'l'
    and transformation 2 mapped 'o' to o, the amount of the column changed
    across all steps is: [2, 1], for that particular row. If mode is min,
    we take 1 row that row and the average text transformed for the row is
    20%. This is averaged across all rows.

    Args:
        feature: feature of dataset
        cleaner: callable that will perform all transformations to row of
            feature
        mode: callable operation to apply to list of characters matched for
            each transformation. Defines how we want to average for each row.

    Returns:
        Average amount of each column transformed, where 'mode' of all
        transformations is used to determine how much of the string was
        transformed.

    """
    f = feature
    matched_lens = [
        (
            cleaner(f.at[i, f.columns[0]]).match_lens,
            len(str(f.at[i, f.columns[0]]))
            if len(str(f.at[i, f.columns[0]])) > 0
            else 1,
        )
        for i in f.index
    ]
    return sum(
        [mode(list_lens) / row_len for list_lens, row_len in matched_lens]
    ) / len(feature)


def num_valid(X):
    """Count the number of valid numbers in an input.

    Args:
        X (iterable): Input data

    Returns:
        A proportion of the data that evaluated as an number

    """
    X = check_series(X)
    data = ~X.apply(pd.to_numeric, errors="coerce").isnull()

    return float(data.sum()) / data.size


def unique_heur(X):
    """Compute the ratio of unique numbers to the total size of the input.

    Args:
        X (iterable): Input data

    Returns:
        1 - the proportion of the data that is unique (ie more unique results \
            in a number closer to one)

    """
    X = check_series(X)
    return 1 - (1.0 * X.nunique() / X.count())


def is_numeric(X):
    """Check if an input is numeric.

    Note:
        Uses pandas method, `is_numeric_dtype \
            <pandas.api.types.is_numeric_dtype>` to make determination.

    Args:
        X (iterable): Input data

    Returns:
        return True if the input is a numeric type, False otherwise.

    """
    X = check_series(X)
    return is_numeric_dtype(X)


def is_string(X):
    """Check if an input is a string.

    Note:
        Uses pandas method, `is_numeric_dtype \
            <pandas.api.types.is_string_dtype>` to make determination.

    Args:
        X (iterable): Input data

    Returns:
        return True if the input is a string type, False otherwise.

    """
    X = check_series(X)
    return is_string_dtype(X)


def has_long_text(X):
    """Check if an input has long text, meaning with more than 1 words.

    Args:
        X (iterable): Input data

    Returns:
        A proportion of the data that evaluated as long text.

    """
    X = check_series(X)
    result = X.iloc[:, 0].apply(lambda x: len(x.split()) > 1)
    return sum(result) / X.count()
