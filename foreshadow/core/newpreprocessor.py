"""Defines the Preprocessor step in the Foreshadow DataPreparer pipeline."""

from foreshadow.config import resolve_config
from foreshadow.core.preparerstep import PreparerStep
from foreshadow.core.resolver import Resolver


class Preprocessor(PreparerStep):
    """Apply preprocessing steps to each column.

    Params:
        *args: args to PreparerStep constructor.
        **kwargs: kwargs to PreparerStep constructor.

    """

    # TODO: create column_sharer if not exists in PreparerStep, this is pending
    # Chris's merge so I can take advantage of new core API

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

            transformers_class_list = resolve_config()[intent]["preprocessor"]
            if (transformers_class_list is not None) or (
                len(transformers_class_list) > 0
            ):
                transformer_list = [
                    tc()  # TODO: Allow kwargs in config
                    for tc in transformers_class_list
                ]
            else:
                transformer_list = None  # None or []

            mapping[i] = {"inputs": ([c],), "steps": transformer_list}

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
