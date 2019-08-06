"""Smart FeatureEngineerer for FeatureEngineererMapper step."""
from foreshadow.concrete.internals.notransform import NoTransform

from .smart import SmartTransformer


class FeatureEngineerer(SmartTransformer):
    """Decide which feature engineering function should be applied."""

    def __init__(self, check_wrapped=True, **kwargs):
        super().__init__(check_wrapped=check_wrapped, **kwargs)

    def pick_transformer(self, X, y=None, **fit_params):
        """Get best transformer for a given set of columns.

        Args:
            X: input DataFrame
            y: input labels
            **fit_params: fit_params

        Returns:
            Best feature engineering transformer.

        """
        return NoTransform()
