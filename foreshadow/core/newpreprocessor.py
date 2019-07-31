"""Defines the Preprocessor step in the Foreshadow DataPreparer pipeline."""

# from foreshadow.config import get_intents
from foreshadow.core.preparerstep import PreparerStep
from foreshadow.transformers.core.smarttransformer import SmartTransformer

from foreshadow.core.resolver import Resolver
from foreshadow.config import resolve_config
from foreshadow.utils import get_transformer


class Preprocessor(PreparerStep):
    """Apply preprocessing steps to each column.

    Params:
        *args: args to PreparerStep constructor.
        **kwargs: kwargs to PreparerStep constructor.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_single_pipeline=True, **kwargs)

    def _def_get_mapping(self, X):
        # Add code to auto create column sharer in preparerstep if it is not
        # passed in
        mapping = {}
        for i, c in enumerate(X.columns):
            intent = self.column_sharer["intent", c]
            if intent is None:
                Resolver(column_sharer=self.column_sharer).fit(X)
                intent = self.column_sharer["intent", c]

            transformers_class_list = resolve_config()[intent]['preprocessor']
            transformer_list = [
                tc()  # TODO: Allow kwargs in config
                for tc in transformers_class_list
            ]

            mapping[i] = {
                'inputs': ([c],),
                'steps': transformer_list
            }

        return mapping

    def get_mapping(self, X):
        """Return the mapping of transformations for the DataCleaner step.

        Args:
            X: input DataFrame.

        Returns:
            Mapping in accordance with super.

        """
        return self._def_get_mapping(X)


if __name__ == "__main__":
    from foreshadow.utils.testing import debug
    debug()
    import numpy as np
    import pandas as pd
    from foreshadow.core.column_sharer import ColumnSharer
    columns = ["financials"]
    data = pd.DataFrame({"financials": np.arange(10)}, columns=columns)
    cs = ColumnSharer()
    p = Preprocessor(cs)
    p.fit(data)
    import pdb
    pdb.set_trace()
