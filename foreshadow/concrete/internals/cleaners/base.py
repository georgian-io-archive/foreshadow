"""BaseCleaner for all Cleaner transformers."""

from collections import namedtuple

import pandas as pd

from foreshadow.base import BaseEstimator, TransformerMixin
from foreshadow.exceptions import InvalidDataFrame
from foreshadow.logging import logging
from foreshadow.metrics import (
    MetricWrapper,
    avg_col_regex,
    calculate_percentage_of_rows_matching_regex,
)
from foreshadow.utils import check_df


CleanerReturn = namedtuple("CleanerReturn", ["row", "match_lens"])


def return_original_row(x):  # noqa: D401
    """Method that returns the row as is.

    Args:
        x: a row of data

    Returns:
        the row itself untouched.

    """
    return x


class BaseCleaner(BaseEstimator, TransformerMixin):
    """Base class for any Cleaner Transformer."""

    def __init__(
        self,
        transformations,
        output_columns=None,
        confidence_computation=None,
        default=return_original_row,
        # cache_manager=None,
    ):
        """Construct any cleaner/flattener.

        Args:
            transformations: a callable that takes a string and returns a
                tuple with the length of the transformed characters and then
                transformed string.
            output_columns: If none, any lists returned by the transformations
                are assumed to be separate columns in the new DataFrame.
                Otherwise, pass the names for each desired output
                column to be used.
            confidence_computation: The dict of {metric: weight} for the
                subclass's metric computation. This implies an OVR model.
            default: Function that returns the default value for a row if
                the transformation failed. Accepts the row as input.

        Raises:
            ValueError: If not a list, int, or None specifying expected
                output columns.

        """
        if not isinstance(output_columns, (int, list, type(None))):
            raise ValueError("output columns not a valid type")

        self.default = default
        self.output_columns = output_columns
        self.transformations = transformations
        self.confidence_computation = {
            MetricWrapper(calculate_percentage_of_rows_matching_regex): 0.8,
            MetricWrapper(avg_col_regex): 0.2,
        }
        # self.confidence_computation = {regex_rows: 0.8, avg_col_regex: 0.2}
        # self.cache_manager = cache_manager
        if confidence_computation is not None:
            self.confidence_computation = confidence_computation

    def metric_score(self, X):
        """Compute the score for this cleaner using confidence_computation.

        confidence_computation is passed through init for each subclass.
        The confidence determines which cleaner/flattener is picked in an
        OVR fashion.

        Args:
            X: input DataFrame.

        Returns:
            float: confidence value.

        """
        # TODO can we also do a sampling here?
        logging.debug("Calculating scores....")
        scores = []
        for metric_wrapper, weight in self.confidence_computation.items():
            scores.append(
                metric_wrapper.calculate(X, cleaner=self.transform_row)
                * weight
            )
        logging.debug("End calculating scores...")
        return sum(scores)

    def transform_row(self, row_of_feature, return_tuple=True):
        """Perform clean operations on text, that is a row of feature.

        Uses self.transformations determined at init time by the child class
        and performs the transformations sequentially.

        Args:
            row_of_feature: one row of one column
            return_tuple: return named_tuple object instead of just the row.
                This will often be set to False when passing this method to an
                external function (non source code) that will expect the
                output to only be the transformed row, such as DataFrame.apply.

        Returns:
            NamedTuple object with:
            .text
            the text in row_of_feature transformed by transformations. If
            not possible, it will be None.
            .match_lens
            the number of characters from original text at each step that
            was transformed.

        """
        matched_lengths = []  # this does not play nice with creating new
        # columns
        for transform in self.transformations:
            row = row_of_feature
            row, match_len = transform(row)
            if match_len == 0:
                matched_lengths.append(0)
                row = self.default(row_of_feature)
                break
            matched_lengths.append(match_len)
        if return_tuple:
            return CleanerReturn(row, matched_lengths)
        else:
            return row

    def fit(self, X, y=None):
        """Empty fit.

        Args:
            X: input observations
            y: input labels

        Returns:
            self

        """
        return self

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

        Raises:
            InvalidDataFrame: If unexpected output returned that was not
                handled correctly. This happens if the output specified by the
                child does not match what is actually returned. The child
                should ensure it's implementation is consistent.

        """
        X = check_df(X, single_column=True)
        logging.info("Starting cleaning rows...")
        out = X[X.columns[0]].apply(self.transform_row, return_tuple=False)
        logging.info("Ending cleaning rows...")
        # access single column as series and apply the list of
        # transformations to each row in the series.
        if any(
            [
                isinstance(out.iloc[i], (list, tuple))
                for i in range(out.shape[0])
            ]
        ):  # out are lists == new columns
            if not all(
                [
                    len(out.iloc[0]) == len(out.iloc[i])
                    for i in range(len(out.iloc[0]))
                ]
            ):
                raise InvalidDataFrame(
                    "length of lists: {}, returned not of same value.".format(
                        [out.iloc[i] for i in range(len(out[0]))]
                    )
                )
            columns = self.output_columns
            if columns is None:
                # by default, pandas would have given a unique integer to
                # each column, instead, we keep the previous column name and
                # add that integer.
                columns = [
                    X.columns[0] + str(c) for c in range(len(out.iloc[0]))
                ]
            # We need to set the index. Otherwise, the new data frame might
            # misalign with other columns.
            X = pd.DataFrame([*out.values], index=out.index, columns=columns)
        elif any(
            [isinstance(out.iloc[i], (dict)) for i in range(out.shape[0])]
        ):  # out are dicts ==  named new columns
            all_keys = dict()
            for row in out:
                all_keys.update({key: True for key in row})  # get all columns
            columns = list(all_keys.keys())
            out = pd.DataFrame([*out.values], columns=columns)
            out.columns = [X.columns[0] + "_" + c for c in columns]
            X = out
            # by default, this will create a DataFrame where if a row
            # contains the value, it will be added, if not NaN is added.
        else:  # no lists, still 1 column output
            X[X.columns[0]] = out
        return X
