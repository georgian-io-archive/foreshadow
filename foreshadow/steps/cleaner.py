"""Cleaner module for handling the cleaning and shaping of data."""
from typing import NoReturn

import pandas as pd

from foreshadow.logging import logging
from foreshadow.smart import Cleaner, Flatten

from .preparerstep import PreparerStep


def _check_empty_columns(X) -> NoReturn:
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
    elif len(empty_columns) > 0:
        logging.info(
            "Identified columns with over 90% missing values: {}"
            "".format(",".join(empty_columns.tolist()))
        )

    return empty_columns.tolist()


class CleanerMapper(PreparerStep):
    """Determine and perform best data cleaning step."""

    def __init__(self, **kwargs):
        """Define the single step for CleanerMapper, using SmartCleaner.

        Args:
            **kwargs: kwargs to PreparerStep constructor.

        """
        self._empty_columns = None
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
        self._empty_columns = _check_empty_columns(Xt)
        return Xt.drop(columns=self._empty_columns)

    def transform(self, X, *args, **kwargs):
        """Clean the dataframe.

        Args:
            X: the data frame.
            *args: positional args.
            **kwargs: key word args.

        Returns:
            A transformed dataframe.

        Raises:
            ValueError: new empty columns detected.

        """
        if not self.has_fitted():
            raise ValueError("Cleaner has not been fitted yet.")

        Xt = super().transform(X, *args, **kwargs)
        return Xt.drop(columns=self._empty_columns)
