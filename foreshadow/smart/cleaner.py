"""SmartCleaner for DataPreparer step."""

from foreshadow.concrete.internals import NoTransform
from foreshadow.config import config
from foreshadow.logging import logging
from foreshadow.utils import AcceptedKey, ConfigKey, DataSamplingMixin

from .smart import SmartTransformer


class Cleaner(SmartTransformer, DataSamplingMixin):
    """Intelligently decide which cleaning function should be applied."""

    def __init__(
        self,  # manually adding as otherwise get_params won't see it.
        check_wrapped=False,
        **kwargs
    ):
        self.single_input = True  # all transformers under this only accept
        # 1 column. This is how DynamicPipeline knows this.
        # get_params then set_params, it may be in kwargs already
        super().__init__(check_wrapped=check_wrapped, **kwargs)

    def pick_transformer(self, X, y=None, **fit_params):
        """Get best transformer for a given column.

        Args:
            X: input DataFrame
            y: input labels
            **fit_params: fit_params

        Returns:
            Best data cleaning transformer.

        """
        # TODO do we want to parallize this step?
        cleaners = config.get_cleaners(cleaners=True)

        user_provided_cleaners = self.cache_manager[
            AcceptedKey.CUSTOMIZED_TRANSFORMERS
        ][ConfigKey.CUSTOMIZED_CLEANERS]
        if len(user_provided_cleaners) > 0:
            cleaners.extend(user_provided_cleaners)

        best_score = 0
        best_cleaner = None
        logging.debug("Picking cleaners...")

        # The sampling is to speed up the metric score calculation as it may
        # not be necessary to scan every row in the data frame to generate a
        # score.
        sampled_df = self.sample_data_frame(df=X)

        # TODO if this improvement is not sufficient, we can try using
        #  multiprocessing to get the scores instead of doing it sequentially.
        for cleaner in cleaners:
            cleaner = cleaner()
            score = cleaner.metric_score(sampled_df)
            if score > best_score:
                best_score = score
                best_cleaner = cleaner
        if best_cleaner is None:
            return NoTransform()
        logging.debug("Picked...")
        return best_cleaner

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
