"""Resolver module that computes the intents for input data."""

from foreshadow.smart.intent_resolving import IntentResolver

from .preparerstep import PreparerStep


class IntentMapper(PreparerStep):
    """Apply intent resolution to each column.

    Params:
        *args: args to PreparerStep constructor.
        **kwargs: kwargs to PreparerStep constructor.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_mapping(self, X):
        """Return the mapping of transformations for the CleanerMapper step.

        Args:
            X: input DataFrame.

        Returns:
            Mapping in accordance with super.

        """
        return self.separate_cols(
            transformers=[
                [IntentResolver(cache_manager=self.cache_manager)] for c in X
            ],
            cols=X.columns,
        )
