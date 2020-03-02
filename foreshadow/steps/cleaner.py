"""Cleaner module for handling the cleaning and shaping of data."""
from typing import List

from foreshadow.ColumnTransformerWrapper import ColumnTransformerWrapper
from foreshadow.concrete import DropCleaner
from foreshadow.logging import logging
from foreshadow.smart import Cleaner
from foreshadow.utils import AcceptedKey, ConfigKey

from .preparerstep import PreparerStep


# def _check_empty_columns(X) -> NoReturn:
#     """Check if all columns are empty in the dataframe.
#
#     Args:
#         X: the dataframe
#
#     Returns:
#         the empty columns.
#
#     Raises:
#         ValueError: all columns are dropped.
#
#     """
#     columns = pd.Series(X.columns)
#     empty_columns = columns[X.isnull().all(axis=0).values]
#
#     if len(empty_columns) == len(columns):
#         error_message = (
#             "All columns are dropped since they all have "
#             "over 90% of missing values. Aborting foreshadow."
#         )
#         logging.error(error_message)
#         raise ValueError(error_message)
#     elif len(empty_columns) > 0:
#         logging.info(
#             "Identified columns with over 90% missing values: {}"
#             "".format(",".join(empty_columns.tolist()))
#         )
#
#     return empty_columns.tolist()


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
        columns = X.columns
        list_of_tuples = [
            (
                column,
                # make_pipeline(
                #     Flatten(cache_manager=self.cache_manager),
                #     Cleaner(cache_manager=self.cache_manager),
                # ),
                Cleaner(cache_manager=self.cache_manager),
                column,
            )
            for column in columns
        ]
        self.feature_processor = ColumnTransformerWrapper(
            list_of_tuples,
            n_jobs=self.cache_manager[AcceptedKey.CONFIG][ConfigKey.N_JOBS],
        )
        self.feature_processor.fit(X=X)
        self._empty_columns = self._check_empty_columns(
            original_columns=columns
        )
        return self

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

    def fit_transform(self, X, *args, **kwargs):
        """Fit then transform the cleaner step.

        Args:
            X: the data frame.
            *args: positional args.
            **kwargs: key word args.

        Returns:
            A transformed dataframe.

        """
        return self.fit(X, *args, **kwargs).transform(X)
        # Xt = super().fit_transform(X, *args, **kwargs)
        # self._empty_columns = _check_empty_columns(Xt)
        # return Xt.drop(columns=self._empty_columns)

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

        # Xt = super().transform(X, *args, **kwargs)

        Xt = self.feature_processor.transform(X=X)
        # Xt = pd.DataFrame(data=Xt, columns=X.columns)
        return Xt.drop(columns=self._empty_columns)
