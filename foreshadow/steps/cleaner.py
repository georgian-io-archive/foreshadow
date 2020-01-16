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
            "Dropping columns due to missing values over 90%: {}"
            "".format(",".join(empty_columns.tolist()))
        )

    return empty_columns.tolist()


def _abort_if_has_new_empty_columns(current_columns, empty_columns):
    new_empty_columns = []
    for column in empty_columns:
        if column in current_columns:
            new_empty_columns.append(column)
    if len(new_empty_columns) > 0:
        raise ValueError(
            "Found new empty columns not present in the training "
            "data. Downstream steps will fail: {}".format(new_empty_columns)
        )


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
        self.empty_columns = _check_empty_columns(Xt)
        return Xt.drop(columns=self.empty_columns)

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
        if self.empty_columns is None:
            raise ValueError("Cleaner has not been fitted yet.")

        Xt = super().transform(X, *args, **kwargs)
        empty_columns_from_transformed_dataset = _check_empty_columns(Xt)

        Xt_after_dropping_columns_identified_during_training = Xt.drop(
            columns=self.empty_columns
        )

        _abort_if_has_new_empty_columns(
            list(Xt_after_dropping_columns_identified_during_training.columns),
            empty_columns_from_transformed_dataset,
        )
        return Xt_after_dropping_columns_identified_during_training
