"""Cleaner module for cleaning data as step in Foreshadow workflow."""
from collections import namedtuple
from sklearn.base import BaseEstimator, TransformerMixin

from foreshadow.core.base import PreparerStep
from foreshadow.exceptions import SmartResolveError
from foreshadow.metrics.internals import avg_col_regex, regex_rows
from foreshadow.transformers.smart import SmartTransformer
from foreshadow.utils.testing import dynamic_import
from foreshadow.transformers.transformers import make_pandas_transformer

import pandas as pd


RegexReturn = namedtuple('RegexReturn', ['text', 'match_lens'])


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
        print(self.separate_cols([(SmartCleaner(), col) for col in X]))
        return self.separate_cols([(SmartCleaner(), col) for col in X])


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

        cleaners = [dynamic_import(cleaner, 'foreshadow.cleaners.internals')
                    for cleaner in cleaners]
        best_score = 0
        best_cleaner = None
        for cleaner in cleaners:
            cleaner = cleaner()
            score = cleaner.metric_score(X)
            if score > best_score:
                best_score = score
                best_cleaner = cleaner

        if best_cleaner is None:
            raise SmartResolveError("best cleaner could not be determined.")
        self.transformer = best_cleaner


class BaseCleaner(BaseEstimator, TransformerMixin):
    """Base class for any Cleaner Transformer."""

    def __init__(self, transformations, confidence_computation=None):
        """

        Args:
            transformations: a callable that takes a string and returns a
            tuple with the length of the transformed characters and then
            transformed string.
            confidence_computation:
        """
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
            text = row_of_feature
            text, search_result = transform(text, return_search=True)
            if search_result is None:
                return RegexReturn(row_of_feature, 0)
            matched_lengths.append(search_result)
        return RegexReturn(text, matched_lengths)

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
        """Clean string columns to prepare for financial transformer.

        Args:
            X (:obj:`pandas.Series`): X data
            y: input labels

        Returns:
            :obj:`pandas.DataFrame`: Transformed data

        """
        X = X.copy()
        # for col in X:  # TODO make this better using .apply.
        #     for transform in self.transformations:
        #         x_col = X[col].apply(transform(X[col]), axis=1)
        #     new_x.join()
        # assume single column

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

        for index, row in X[X.columns[0]].iterrows():
            df
        X[X.columns[0]] = [self(row).text for row in X[X.columns[0]]]