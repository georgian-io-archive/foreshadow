"""Resolver module that computes the intents for input data."""

from collections import namedtuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from foreshadow.core.preparerstep import PreparerStep
from foreshadow.exceptions import InvalidDataFrame
from foreshadow.metrics.internals import avg_col_regex, regex_rows
from foreshadow.transformers.core import SmartTransformer
from foreshadow.transformers.core.notransform import NoTransform
from foreshadow.transformers.core.smarttransformer import SmartTransformer
from foreshadow.utils.testing import dynamic_import
from foreshadow.utils.validation import check_df

from foreshadow.config import get_intents

from abc import ABCMeta, abstractmethod

class IntentResolver(PreparerStep):
    """Apply intent resolution to each column."""

    def __init__(self, *args, **kwargs):
        """Define the single step for DataCleaner, using SmartCleaner.

        Args:
            *args: args to PreparerStep constructor.
            **kwargs: kwargs to PreparerStep constructor.

        """
        super().__init__(*args, use_single_pipeline=True, **kwargs)

    def get_mapping(self, X):
        """Return the mapping of transformations for the DataCleaner step.

        Args:
            X: input DataFrame.

        Returns:
            Mapping in accordance with super.

        """
        return self.separate_cols(
            transformers=[
                [Resolver(column_sharer=self.column_sharer)]
                for c in X
            ],
            X=X,
        )

class Resolver(SmartTransformer):
    """Determine the intent for a particular column

    Args:
        column_sharer (`foreshadow.core.column_sharer.ColumnSharer`): shared

    """
    validate_wrapped = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def resolve_intent(self, X, y=None):
        return max(get_intents(), key=lambda intent: intent.get_confidence(X))


    def pick_transformer(self, X, y=None, **fit_params):
        intent_class = self.resolve_intent(X, y=y)

        column_name = X.columns[0]
        self.column_sharer["intent", column_name] = intent_class.__name__

        return intent_class()


if __name__ == "__main__":
    from foreshadow.utils.testing import debug

    debug()

    import numpy as np
    import pandas as pd
    from foreshadow.cleaners import DataCleaner
    from foreshadow.core.column_sharer import ColumnSharer

    columns = ["financials"]
    data = pd.DataFrame({"financials": np.arange(10)}, columns=columns)
    cs = ColumnSharer()
    ir = IntentResolver(cs)
    ir.fit(data)

    import pdb

    pdb.set_trace()
