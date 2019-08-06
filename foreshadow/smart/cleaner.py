"""SmartCleaner for DataPreparer step."""

from foreshadow.concrete.internals import NoTransform
from foreshadow.config import config
from foreshadow.logging import logging

from .smart import SmartTransformer


class Cleaner(SmartTransformer):
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
        cleaners = config.get_cleaners(cleaners=True)
        best_score = 0
        best_cleaner = None
        for cleaner in cleaners:
            cleaner = cleaner()
            score = cleaner.metric_score(X)
            if score > best_score:
                best_score = score
                best_cleaner = cleaner
        if best_cleaner is None:
            return NoTransform()
        return best_cleaner

    def resolve(self, X, *args, **kwargs):
        """Resolve the underlying concrete transformer.

        Sets self.column_sharer with the domain tag.

        Args:
            X: input DataFrame
            *args: args to super
            **kwargs: kwargs to super

        Returns:
            Return from super.

        """
        ret = super().resolve(X, *args, **kwargs)
        if self.column_sharer is not None:
            self.column_sharer[
                "domain", X.columns[0]
            ] = self.transformer.__class__.__name__
        else:
            logging.debug("column_sharer was None")
        return ret
