"""Cleaner module for cleaning data as step in Foreshadow workflow."""
from collections import namedtuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from foreshadow.core.base import PreparerStep
from foreshadow.exceptions import InvalidDataFrame, SmartResolveError
from foreshadow.metrics.internals import avg_col_regex, regex_rows
from foreshadow.transformers.smart import SmartTransformer
from foreshadow.transformers.transformers import (
    _Empty,
    make_pandas_transformer,
)
from foreshadow.utils.testing import dynamic_import
from foreshadow.utils.validation import check_df


CleanerReturn = namedtuple("CleanerReturn", ["row", "match_lens"])


@make_pandas_transformer
class DataCleaner(PreparerStep):
    """Determine and perform best data cleaning step."""

    def __init__(self, *args, **kwargs):
        """Define the single step for DataCleaner, using SmartCleaner.

        Args:
            *args: args to PreparerStep constructor.
            **kwargs: kwargs to PreparerStep constructor.

        """
        super().__init__(*args, **kwargs)

    def get_mapping(self, X):
        return self.separate_cols(
            transformers=[[SmartFlatten(), SmartCleaner()] for col in X], X=X
        )


class SmartCleaner(SmartTransformer):
    """Intelligently decide which cleaning function should be applied."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def pick_transformer(self, X, y=None, **fit_params):
        """Get best transformer for a given column.

        Args:
            X: input DataFrame
            y: input labels
            **fit_params: fit_params

        Returns:
            Best data cleaning transformer.

        """
        from foreshadow.cleaners.internals import __all__ as cleaners

        cleaners = [
            dynamic_import(cleaner, "foreshadow.cleaners.internals")
            for cleaner in cleaners
            if cleaner.lower().find("cleaner")
        ]
        best_score = 0
        best_cleaner = None
        for cleaner in cleaners:
            cleaner = cleaner()
            score = cleaner.metric_score(X)
            if score > best_score:
                best_score = score
                best_cleaner = cleaner

        if best_cleaner is None:
            self.transformer = _Empty()
        else:
            self.transformer = best_cleaner


class SmartFlatten(SmartTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def pick_transformer(self, X, y=None, **fit_params):
        """Get best transformer for a given column.

        Args:
            X: input DataFrame
            y: input labels
            **fit_params: fit_params

        Returns:
            Best data flattening transformer

        """
        from foreshadow.cleaners.internals import __all__ as cleaners

        flatteners = [
            dynamic_import(cleaner, "foreshadow.cleaners.internals")
            for cleaner in cleaners
            if cleaner.lower().find("flatten")
        ]

        best_score = 0
        best_flattener = None
        for flattener in flatteners:
            flattener = flattener()
            score = flattener.metric_score(X)
            if score > best_score:
                best_score = score
                best_flattener = flattener

        if best_flattener is None:
            self.transformer = _Empty()
        else:
            self.transformer = best_flattener


class BaseCleaner(BaseEstimator, TransformerMixin):
    """Base class for any Cleaner Transformer."""

    def __init__(
        self, transformations, output_columns=None, confidence_computation=None
    ):
        """

        Args:
            transformations: a callable that takes a string and returns a
                tuple with the length of the transformed characters and then
                transformed string.
            output_columns: If none, any lists returned by the transformations
                are assumed to be separate columns in the new DataFrame.
                Otherwise, pass the names for each desired output
                column to be used.
            confidence_computation:
        """
        if not isinstance(output_columns, (int, list, None)):
            raise ValueError("output columns not a valid type")

        self.output_columns = output_columns
        self.transformations = transformations
        self.confidence_computation = {regex_rows: 0.8, avg_col_regex: 0.2}
        if confidence_computation is not None:
            self.confidence_computation = confidence_computation

    def metric_score(self, X):
        """Compute the score for this cleaner using

        Args:
            X:

        Returns:

        """
        return sum(
            [
                metric_fn(X, encoder=self) * weight
                for metric_fn, weight in self.confidence_computation.items()
            ]
        )

    def __call__(self, row_of_feature):
        """Perform clean operations on text, that is a row of feature.

        By

        Args:
            row_of_feature: one row of one column

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
            row, match_len = transform(row, return_search=True)
            if match_len == 0:
                return CleanerReturn(row_of_feature, 0)
            matched_lengths.append(match_len)
        return CleanerReturn(row, matched_lengths)

    def fit(self, X, y=None):
        """Empty fit.

        Args:
            X: input observations
            y: input labels

        Returns:
            self

        """
        if isinstance(self.transformer, _Empty):
            return None
        else:
            return self

    def transform(self, X, y=None):
        """Clean string columns to prepare for financial transformer.

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
        # PRoblem:
        # I can use .apply to perform all these transformations and that
        # works beautifully, except when I want to define a funtion that
        # will use the pandas.series.str.split operation. In which case,
        # the .apply fails and I don't know why.

        # I need each function to accept the row as an argument so that we
        # can inspect how much of the text was matched (for determining if
        # it should be used). however, doing this means I need to iterate
        # over each row for a given column on my own, which requires me to
        # leave

        outputs = X[X.columns[0]].apply(self)  # access single column as
        # series and apply the list of transformations to each row in the
        # series.
        if any(
            [
                isinstance(outputs[i], (list, tuple))
                for i in range(outputs.shape[0])
            ]
        ):  # outputs are lists == new columns
            if not all(
                [len(X[0]) == len(X[i]) for i in range(outputs.shape[0])]
            ):
                raise InvalidDataFrame(
                    "length of lists returned not of same " "value."
                )
            if self.output_columns is None:
                columns = self.output_columns
                if columns is None:
                    columns = []
                X = pd.DataFrame.from_items(
                    zip(outputs.index, outputs.values), columns=columns
                ).T
        elif any(
            [isinstance(outputs[i], (dict)) for i in range(outputs.shape[0])]
        ):  # outputs are dicts ==  named new columns
            all_keys = dict()
            for row in outputs:
                all_keys.update({key: True for key in row})  # get all columns
            X = pd.DataFrame(outputs.values, columns=list(all_keys.keys()))
            # by default, this will create a DataFrame where if a row
            # contains the value, it will be added, if not NaN is added.
        else:  # no lists, still 1 column output
            X[X.columns[0]] = outputs
        return X
