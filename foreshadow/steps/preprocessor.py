"""Defines the Preprocessor step in the Foreshadow DataPreparer pipeline."""

from foreshadow.config import config
from foreshadow.utils import Override

from .autointentmap import AutoIntentMixin
from .preparerstep import PreparerMapping, PreparerStep


class Preprocessor(PreparerStep, AutoIntentMixin):
    """Apply preprocessing steps to each column.

    Params:
        *args: args to PreparerStep constructor.
        **kwargs: kwargs to PreparerStep constructor.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def check_process(self, X):
        """Check Process and please see the parent class.

        Args:
            X: The data frame being processed

        """
        super().check_process(X)
        # This is to pass a few unit tests, which we may we want to rewrite.
        self.configure_cache_manager(cache_manager=self.cache_manager)

    def _handle_intent_override(self, default_parallel_process):
        for i in range(len(default_parallel_process.transformer_list)):
            name, trans, cols = default_parallel_process.transformer_list[i]
            column = cols[0]  # Preprocessor is per column based transformation
            override_key = "_".join([Override.INTENT, column])
            if (
                self.cache_manager.has_override()
                and override_key in self.cache_manager["override"]
            ):
                self._parallel_process.transformer_list[
                    i
                ] = default_parallel_process.transformer_list[i]

    def get_mapping(self, X):
        """Return the mapping of transformations for the DataCleaner step.

        Args:
            X: input DataFrame.

        Returns:
            Mapping in accordance with super.

        """
        pm = PreparerMapping()
        for i, c in enumerate(X.columns):
            self.check_resolve(X)
            intent = self.cache_manager["intent", c]
            transformers_class_list = config.get_preprocessor_steps(intent)
            if (transformers_class_list is not None) or (
                len(transformers_class_list) > 0
            ):
                transformer_list = [
                    # tc(cache_manager=self.cache_manager)
                    tc()
                    # TODO: Allow kwargs in config
                    for tc in transformers_class_list
                ]
            else:
                transformer_list = None  # None or []
            pm.add([c], transformer_list, i)
        return pm
