"""Cleaner module for handling the cleaning and shaping of data."""
from typing import NoReturn

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
        self.empty_columns = None
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

        """
        Xt = super().fit_transform(X, *args, **kwargs)
        self.empty_columns = self._check_empty_dataframe(Xt)
        return Xt.drop(columns=self.empty_columns)

    def transform(self, X, *args, **kwargs):
        """Clean the dataframe.

        Args:
            X: the data frame.
            *args: positional args.
            **kwargs: key word args.

        Returns:
            A transformed dataframe.

        """
        Xt = super().transform(X, *args, **kwargs)
        # if self.empty_columns is None:
        #     self.empty_columns = self._check_empty_dataframe(Xt)
        return Xt.drop(columns=self.empty_columns)

    def _check_empty_dataframe(self, X) -> NoReturn:
        """Check if all columns are empty in the dataframe.

        Args:
            X: the dataframe

        Returns:
            the empty columns.

        Raises:
            ValueError: all columns are dropped.

        """
        columns = pd.Series(X.columns)
        empty_columns = columns[X.isnull().all(axis=0).values]

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

        return empty_columns
