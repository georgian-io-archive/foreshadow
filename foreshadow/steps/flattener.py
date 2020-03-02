"""Cleaner module for handling the flattening of the data."""

from foreshadow.ColumnTransformerWrapper import ColumnTransformerWrapper
from foreshadow.smart import Flatten
from foreshadow.utils import AcceptedKey, ConfigKey

from .preparerstep import PreparerStep


class FlattenMapper(PreparerStep):
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
            (column, Flatten(cache_manager=self.cache_manager), column)
            for column in columns
        ]
        self.feature_processor = ColumnTransformerWrapper(
            list_of_tuples,
            n_jobs=self.cache_manager[AcceptedKey.CONFIG][ConfigKey.N_JOBS],
        )
        self.feature_processor.fit(X=X)
        return self

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

        """
        # if self._empty_columns is None:
        #     raise ValueError("Cleaner has not been fitted yet.")

        # Xt = super().transform(X, *args, **kwargs)

        Xt = self.feature_processor.transform(X=X)
        return Xt
        # Xt = pd.DataFrame(data=Xt, columns=X.columns)
        # return Xt.drop(columns=self._empty_columns)
