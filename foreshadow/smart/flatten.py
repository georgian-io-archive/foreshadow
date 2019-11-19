"""SmartFlattener for the CleanerMapper step in DataPreparer."""
from foreshadow.concrete.internals import NoTransform
from foreshadow.config import config
from foreshadow.logging import logging

from .smart import SmartTransformer


class Flatten(SmartTransformer):
    """Smartly determine how to flatten an input DataFrame."""

    def __init__(self, check_wrapped=True, **kwargs):
        super().__init__(check_wrapped=check_wrapped, **kwargs)

    def pick_transformer(self, X, y=None, **fit_params):
        """Get best transformer for a given column.

        Args:
            X: input DataFrame
            y: input labels
            **fit_params: fit_params

        Returns:
            Best data flattening transformer

        """
        flatteners = config.get_cleaners(flatteners=True)
        best_score = 0
        best_flattener = None
        for flattener in flatteners:
            flattener = flattener()
            score = flattener.metric_score(X)
            if score > best_score:
                best_score = score
                best_flattener = flattener
        if best_flattener is None:
            return NoTransform()
        return best_flattener

    def should_force_reresolve_based_on_override(self, X):
        """Check if it should force reresolve based on user override.

        Args:
            X: the data frame

        Returns:
            bool: whether we should force reresolve based on user override.

        """
        # TODO we do not want data cleaners to force reresolve because of
        #  intent override. We will implement proper override handling for
        #  them in the future. For now, disable by returning False.
        return False

    def resolve(self, X, *args, **kwargs):
        """Resolve the underlying concrete transformer.

        Sets self.cache_manager with the domain tag.

        Args:
            X: input DataFrame
            *args: args to super
            **kwargs: kwargs to super

        Returns:
            Return from super.

        """
        ret = super().resolve(X, *args, **kwargs)
        if self.cache_manager is not None:
            self.cache_manager[
                "domain", X.columns[0]
            ] = self.transformer.__class__.__name__
        else:
            logging.debug("cache_manager was None")
        return ret
