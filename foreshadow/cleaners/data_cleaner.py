"""Cleaner module for cleaning data as step in Foreshadow workflow."""
from collections import namedtuple
from sklearn.base import BaseEstimator, TransformerMixin

from foreshadow.core.base import PreparerStep
from foreshadow.exceptions import SmartResolveError
from foreshadow.metrics.internals import avg_col_regex, regex_rows
from foreshadow.transformers.smart import SmartTransformer
from foreshadow.utils.testing import dynamic_import
from foreshadow.transformers.transformers import make_pandas_transformer


RegexReturn = namedtuple('RegexReturn', ['text', 'search_result'])


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
        """

        Args:
            row_of_feature:

        Returns:

        """
        search_results = []
        for transform in self.transformations:
            text = row_of_feature
            text, search_result = transform(text, return_search=True)
            if search_result is None:
                return None, row_of_feature
            search_results.append(search_result)
        return RegexReturn(text, search_results)
