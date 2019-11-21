"""Class defintion for BaseFeaturizeViaLambda abstract class."""

from typing import Callable

import pandas as pd

from .base_featurizer import BaseFeaturizer


class BaseFeaturizerViaLambda(BaseFeaturizer):
    """
    Abstract class to create secondary featurization via a custom function.

    Attributes:
        _callable {Callable[[pd.DataFrame], pd.Series]}
            -- User-defined function to extract metafeatures from dataframe.
        sec_feature_names {List[str]}
            -- Names for secondary metafeatures

        Refer to superclass for additional attributes.
    """

    def __init__(
        self,
        method: str,
        callable_: Callable[[pd.DataFrame], pd.Series],
        normalizable: bool,
    ):
        """
        Init function.

        Extends superclass method.

        Arguments:
            method {str}
                -- Description of secondary metafeatures. Used in naming
                   `sec_feature_names`.
            callable_ {Callable[[pd.DataFrame], pd.Series]}
                -- User-defined function to extract metafeatures from dataframe.
            normalizable {bool}
                -- Whether the generated feature should be normalized.
        """
        super().__init__(method=method, normalizable=normalizable)
        self._callable = callable_
        self.sec_feature_names = super()._mark_nonnormalizable(
            [self.method], normalizable=self.normalizable
        )
