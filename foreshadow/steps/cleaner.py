"""Cleaner module for handling the cleaning and shaping of data."""
import pandas as pd

from foreshadow.logging import logging
from foreshadow.smart import Cleaner, Flatten

from .preparerstep import PreparerStep


class CleanerMapper(PreparerStep):
    """Determine and perform best data cleaning step."""

    def __init__(self, **kwargs):
        """Define the single step for CleanerMapper, using SmartCleaner.

        Args:
            **kwargs: kwargs to PreparerStep constructor.

        """
        super().__init__(**kwargs)

    def get_mapping(self, X):
        """Return the mapping of transformations for the CleanerMapper step.

        Args:
            X: input DataFrame.

        Returns:
            Mapping in accordance with super.

        """
        return self.separate_cols(
            transformers=[
                [
                    Flatten(cache_manager=self.cache_manager),
                    Cleaner(cache_manager=self.cache_manager),
                ]
                for c in X
            ],
            cols=X.columns,
        )

    def fit_transform(self, X, *args, **kwargs):
        """Fit then transform the cleaner step.

        Args:
            X: the data frame.
            *args: positional args.
            **kwargs: key word args.

        Returns:
            A transformed dataframe.

        Raises:
            ValueError: all columns are dropped.

        """
        Xt = super().fit_transform(X, *args, **kwargs)
        columns = pd.Series(Xt.columns)
        empty_columns = columns[Xt.isnull().all(axis=0).values]

        if len(empty_columns) == len(columns):
            error_message = (
                "All columns are dropped since they all have "
                "over 90% of missing values. Aborting foreshadow."
            )
            logging.error(error_message)
            raise ValueError(error_message)
        else:
            logging.info(
                "Dropping columns due to missing values over 90%: "
                "".format(",".join(empty_columns.tolist()))
            )

        return Xt.dropna(axis=1, how="all")
