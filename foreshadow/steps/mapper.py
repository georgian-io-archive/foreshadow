"""Resolver module that computes the intents for input data."""

from foreshadow.smart.intent_resolving import IntentResolver
from foreshadow.utils import AcceptedKey

from .preparerstep import PreparerStep


class IntentMapper(PreparerStep):
    """Apply intent resolution to each column.

    Params:
        *args: args to PreparerStep constructor.
        **kwargs: kwargs to PreparerStep constructor.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, *args, **kwargs):
        """Fit this step.

        calls underlying parallel process.

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
        self._update_cache_manager_with_intents()

        return self

    def _update_cache_manager_with_intents(self):
        for intent_resolver_tuple in self.feature_processor.transformers_:
            intent_resolver = intent_resolver_tuple[1]
            column_name = intent_resolver_tuple[2]
            self.cache_manager[AcceptedKey.INTENT][
                column_name
            ] = intent_resolver.column_intent

    def _construct_column_transformer_tuples(self, X):
        columns = X.columns
        list_of_tuples = [
            (
                column + "_" + IntentMapper.__class__.__name__,
                IntentResolver(
                    column=column, cache_manager=self.cache_manager
                ),
                column,
            )
            for column in columns
        ]
        return list_of_tuples
