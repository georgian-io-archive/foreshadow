"""Cleaner module for handling the flattening of the data."""

from foreshadow.smart import Flatten

from .preparerstep import PreparerStep


class FlattenMapper(PreparerStep):
    """Determine and perform best data flattening step."""

    def __init__(self, **kwargs):
        """Define the single step for FlattenMapper, using SmartCleaner.

        Args:
            **kwargs: kwargs to PreparerStep constructor.

        """
        self._empty_columns = None
        super().__init__(**kwargs)

    def fit(self, X, *args, **kwargs):
        """Fit the flatten step.

        Calls underlying feature processor. It will flatten columns with
        JSON like data but will not touch other columns.

        Args:
            X: input DataFrame
            *args: args to _fit
            **kwargs: kwargs to _fit

        Returns:
            transformed data handled by Pipeline._fit

        """
        list_of_tuples = self._construct_column_transformer_tuples(X=X)
        self._prepare_feature_processor(list_of_tuples=list_of_tuples)
        self.feature_processor.fit(X=X)
        return self

    def _construct_column_transformer_tuples(self, X):
        columns = X.columns
        list_of_tuples = [
            (
                column + "_" + FlattenMapper.__class__.__name__,
                Flatten(cache_manager=self.cache_manager),
                column,
            )
            for column in columns
        ]
        return list_of_tuples
