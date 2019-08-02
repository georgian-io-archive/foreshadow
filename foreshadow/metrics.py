"""Metrics used across Foreshadow for smart decision making."""

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

from foreshadow.logging import logging
from foreshadow.utils import check_series


class MetricWrapper:
    """MetricWrapper class for metric functions.

    Note:
        Apply this class by using the metric decorator.

    Params:
        fn: Metric function to be wrapped
        default_return (bool): The default return value of the wrapped
            function.

    .. automethod:: __call__

    """

    def __init__(self, fn, default_return=None):
        self.fn = fn
        self.default_return = default_return

    def __call__(self, feature, invert=False, **kwargs):
        """Use the metric function passed at initialization.

        Note:
            If default_return was set, the wrapper will suppress any errors
            raised by the wrapped function.

        Args:
            feature: feature/column of pandas dataset
                requires it.
            invert (bool): Invert the output (1-x)
            **kwargs: any keyword arguments to metric function

        Returns:
            The metric computation defined by the metric.

        Raises:
            re_raise: If default return is not set the metric will display \
                the raised errors in the function.

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

        return self._last_call if not invert else (1.0 - self._last_call)

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


class metric:
    """Decorate any metric function.

    Args:
        fn: function to decorate. (Automatically passed in)
        default_return (bool): The default return value of the Metric function.

    Returns:
        Metric function as callable object.

    """

    def __init__(self, default_return=None):
        self.default_return = default_return

    def __call__(self, fn):
        """Get the wrapped metric function.

        Args:
            fn: The metric function to be wrapped.

        Returns:
            An instance `MetricWrapper` that wraps a function.

        """
        return MetricWrapper(fn, self.default_return)


@metric()
def unique_count(feature):
    """Count number of unique values in feature.

    Args:
        feature: feature/column of pandas dataset

    Returns:
        number of unique values in feature as int.

    """
    return len(feature.value_counts())


@metric()
def unique_count_bias(feature):
    """Difference of count of unique values relative to the length of feature.

    Args:
        feature: feature/column of pandas dataset

    Returns:
        Number of unique values relative to the length of dataset.

    """
    return len(feature) - len(feature.value_counts())


@metric()
def unique_count_weight(feature):
    """Normalize count number of unique values relative to length of feature.

    Args:
        feature: feature/column of pandas dataset

    Returns:
        Normalized Number of unique values relative to length of feature.

    """
    return len(feature.value_counts()) / len(feature)


@metric()
def regex_rows(feature, cleaner):
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
    matched_lens = [
        cleaner(f.get_value(i, f.columns[0])).match_lens for i in f.index
    ]
    return sum([min(list_lens) for list_lens in matched_lens]) / len(feature)


@metric()
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
        (cleaner(f.get_value(i, f.columns[0])).match_lens, len(f.iloc[i]))
        for i in f.index
    ]
    return sum(
        [mode(list_lens) / row_len for list_lens, row_len in matched_lens]
    ) / len(feature)


@metric(default_return=0)
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


@metric(default_return=0)
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


@metric(default_return=0)
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


@metric(default_return=0)
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
