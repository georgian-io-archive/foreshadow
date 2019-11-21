"""
Definitions for classes that extract metafeatures from the metafeatures dataframe.

Classes here take a user-defined function to apply to dataframes at the
(test) metafeatures attributes of DataSetParser subclasses.

Class definitions included here are for:
    -- MetaDataSetFeaturizerViaLambda
    -- MetaDataSetFeaturizerViaLambdaBuilder
"""

from typing import Callable, Optional, Union

import pandas as pd

from .base_featurizer_via_lambda import BaseFeaturizerViaLambda


class MetaDataSetFeaturizerViaLambda(BaseFeaturizerViaLambda):
    """
    Create new secondary metafeatures from the (test) metafeatures dataframes.

    Attributes:
        method {str}
            -- Description of featurizer. Used in naming
                `sec_feature_names`.
        callable_ {Callable[[pd.DataFrame],Union[pd.DataFrame, pd.Series]]}
            -- User-defined function to apply on (test) metafeatures DataFrame.
        normalizable {bool}
            -- Whether the generated feature should be normalized.
    """

    def __init__(
        self,
        method: str,
        callable_: Callable[[pd.DataFrame], Union[pd.DataFrame, pd.Series]],
        normalizable: bool,
    ):
        """
        Init function.

        Arguments:
            method {str}
                -- Description of featurizer. Used in naming
                   `sec_feature_names`.
            callable_ {Callable[[pd.DataFrame],Union[pd.DataFrame, pd.Series]]}
                -- User-defined function to apply on (test) metafeatures DataFrame.
            normalizable {bool}
                -- Whether the generated feature should be normalized.
        """
        super().__init__(
            method=method, callable_=callable_, normalizable=normalizable
        )

    def featurize(
        self,
        meta_df: Optional[pd.DataFrame] = None,
        test_meta_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Create secondary features.

        At least one keyword argument must be provided.

        Keyword Arguments:
            meta_df {Optional[pd.DataFrame]} -- Training metafeatures.
                                                (default: {None})
            test_meta_df {Optional[pd.DataFrame]} -- Test metafeatures.
                                                (default: {None})

        Raises:
            ValueError -- If `meta_df` and `test_meta_df` are both None.
        """
        if meta_df is None and test_meta_df is None:
            raise ValueError("At least one keyword argument must be provided.")

        if meta_df is not None:
            self.sec_metafeatures = pd.DataFrame(self._callable(meta_df))
        if test_meta_df is not None:
            self.sec_test_metafeatures = pd.DataFrame(
                self._callable(test_meta_df)
            )


class MetaDataSetFeaturizerViaLambdaBuilder:
    """Builder class for MetaDataSetFeaturizerViaLambda."""

    def __call__(
        self,
        method: str,
        callable_: Callable[[pd.DataFrame], Union[pd.DataFrame, pd.Series]],
        normalizable: bool,
        **_ignore
    ):
        """Build a MetaDataSetFeaturizerViaLambda based on supplied keyword arguments."""
        return MetaDataSetFeaturizerViaLambda(
            method=method, callable_=callable_, normalizable=normalizable
        )
