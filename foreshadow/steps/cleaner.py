"""Cleaner module for handling the cleaning and shaping of data."""
from typing import List

from foreshadow.ColumnTransformerWrapper import ColumnTransformerWrapper
from foreshadow.concrete import DropCleaner
from foreshadow.logging import logging
from foreshadow.smart import Cleaner
from foreshadow.utils import AcceptedKey, ConfigKey

from .preparerstep import PreparerStep


class CleanerMapper(PreparerStep):
    """Determine and perform best data cleaning step."""

    def __init__(self, **kwargs):
        """Define the single step for CleanerMapper, using SmartCleaner.

        Args:
            **kwargs: kwargs to PreparerStep constructor.

        """
        self._empty_columns = None
        super().__init__(**kwargs)

    def fit(self, X, *args, **kwargs):
        """Fit this step.

        calls underlying parallel process.

        Args:
            X: input DataFrame
            *args: args to _fit
            **kwargs: kwargs to _fit

        Returns:
            transformed data handled by Pipeline._fit

        """
        self._prepare_feature_processor(X=X)
        self.feature_processor.fit(X=X)
        self._empty_columns = self._check_empty_columns(
            original_columns=X.columns
        )
        return self

    def transform(self, X, *args, **kwargs):
        """Clean the dataframe.

        Args:
            X: the data frame.
            *args: positional args.
            **kwargs: key word args.

        Returns:
            A transformed dataframe.

        Raises:
            ValueError: Cleaner has not been fitted.

        """
        if self._empty_columns is None:
            raise ValueError("Cleaner has not been fitted yet.")

        Xt = self.feature_processor.transform(X=X)
        return Xt.drop(columns=self._empty_columns)

    def _prepare_feature_processor(self, X):
        columns = X.columns
        list_of_tuples = [
            (
                column + "_" + Cleaner.__class__.__name__,
                Cleaner(cache_manager=self.cache_manager),
                column,
            )
            for column in columns
        ]
        self.feature_processor = ColumnTransformerWrapper(
            list_of_tuples,
            n_jobs=self.cache_manager[AcceptedKey.CONFIG][ConfigKey.N_JOBS],
        )

    def _check_empty_columns(self, original_columns: List) -> List:
        empty_columns = []
        for cleaner_tuple in self.feature_processor.transformers_:
            _, cleaner, column_name = cleaner_tuple
            if isinstance(cleaner.transformer, DropCleaner):
                empty_columns.append(column_name)

        if len(empty_columns) == len(original_columns):
            error_message = (
                "All columns are dropped since they all have "
                "over 90% of missing values. Aborting foreshadow."
            )
            logging.error(error_message)
            raise ValueError(error_message)
        elif len(empty_columns) > 0:
            logging.info(
                "Identified columns with over 90% missing values: {} and "
                "they will be dropped."
                "".format(",".join(empty_columns))
            )

        return empty_columns
