"""Resolver module that computes the intents for input data."""

from foreshadow.preparer.preparerstep import PreparerStep
from foreshadow.smart import Resolver


class ResolverMapper(PreparerStep):
    """Apply intent resolution to each column.

    Params:
        *args: args to PreparerStep constructor.
        **kwargs: kwargs to PreparerStep constructor.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_mapping(self, X):
        """Return the mapping of transformations for the CleanerMapper step.

        Args:
            X: input DataFrame.

        Returns:
            Mapping in accordance with super.

        """
        return self.separate_cols(
            transformers=[
                [Resolver(column_sharer=self.column_sharer)] for c in X
            ],
            cols=X.columns,
        )
