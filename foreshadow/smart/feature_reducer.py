"""Smart Feature Reducer for FeatureReducerMapper step."""
from foreshadow.concrete.internals import NoTransform

from .smart import SmartTransformer


class FeatureReducer(SmartTransformer):
    """Decide which feature reduction function should be applied."""

    def __init__(
        self,  # manually adding as otherwise get_params won't see it.
        check_wrapped=False,
        **kwargs
    ):
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
