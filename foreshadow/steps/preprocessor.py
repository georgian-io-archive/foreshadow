"""Defines the Preprocessor step in the Foreshadow DataPreparer pipeline."""

from foreshadow.config import config

from .autointentmap import AutoIntentMixin
from .preparerstep import PreparerMapping, PreparerStep


class Preprocessor(PreparerStep, AutoIntentMixin):
    """Apply preprocessing steps to each column.

    Params:
        *args: args to PreparerStep constructor.
        **kwargs: kwargs to PreparerStep constructor.

    """

    # TODO: create column_sharer if not exists in PreparerStep, this is pending
    # Chris's merge so I can take advantage of new core API

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _def_get_mapping(self, X):
        pm = PreparerMapping()
        for i, c in enumerate(X.columns):
            self.check_resolve(X)
            intent = self.column_sharer["intent", c]
            transformers_class_list = config.get_preprocessor_steps(intent)
            if (transformers_class_list is not None) or (
                len(transformers_class_list) > 0
            ):
                transformer_list = [
                    tc()  # TODO: Allow kwargs in config
                    for tc in transformers_class_list
                ]
            else:
                transformer_list = None  # None or []
            pm.add([c], transformer_list, i)
        return pm

    def get_mapping(self, X):
        """Return the mapping of transformations for the DataCleaner step.

        Args:
            X: input DataFrame.

        Returns:
            Mapping in accordance with super.

        """
        return self._def_get_mapping(X)
