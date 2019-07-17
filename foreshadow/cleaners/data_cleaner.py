"""Cleaner module for cleaning data as step in Foreshadow workflow."""
from foreshadow.core.base import PreparerStep
from foreshadow.transformers.externals import StandardScaler  # remove this
from foreshadow.transformers.smart import SmartTransformer


class DataCleaner(PreparerStep):
    """Determine and perform best data cleaning step."""

    def __init__(self, *args, **kwargs):
        """Define the single step for DataCleaner, using SmartCleaner.

        Args:
            *args: args to PreparerStep constructor.
            **kwargs: kwargs to PreparerStep constructor.

        """
        steps = [("cleaning", SmartCleaner)]
        super(DataCleaner, self).__init__(steps, *args, **kwargs)

    def get_transformer_list(self, transformer, X):
        """Return transformer mapping to columns.

        Args:
            transformer: SmartTransformer to use
            X: input DataSet

        Returns:
            transformer: col for each column mapping.

        """
        return self.one_smart_all_cols(transformer, X)[0]  # create a
        # separate SmartCleaner for each column

    def get_transformer_weights(self, transformer, X):
        """Return no transformer_weights.

        Args:
            transformer: SmartTransformer to use
            X: input DataSet

        Returns:
            None

        """
        return self.one_smart_all_cols(transformer, X)[1]  # don't use any
        # weighting.


class SmartCleaner(SmartTransformer):
    """Intelligently decide which cleaning function should be applied."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_transformer(self, X, y=None, **fit_params):
        """Stub.

        Args:
            X: input DataFrame
            y: input labels
            **fit_params: fit_params

        Returns:
            StandardScaler()

        """
        return StandardScaler()
