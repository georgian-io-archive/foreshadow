"""SmartSummarizer for FeatureSummarizerMapper step."""
from foreshadow.concrete.internals import NoTransform
from foreshadow.smart.smart import SmartTransformer


class FeatureSummarizer(SmartTransformer):
    """Empty Smart transformer for feature summary step."""

    def __init__(self, check_wrapped=True, **kwargs):
        super().__init__(check_wrapped=check_wrapped, **kwargs)

    def pick_transformer(self, X, y=None, **fit_params):
        """Get best transformer for a given set of columns.

        Args:
            X: input DataFrame
            y: input labels
            **fit_params: fit_params

        Returns:
            No transformer.

        """
        return NoTransform()
