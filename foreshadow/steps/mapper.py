"""Resolver module that computes the intents for input data."""

from foreshadow.ColumnTransformerWrapper import ColumnTransformerWrapper
from foreshadow.smart.intent_resolving import IntentResolver
from foreshadow.utils import AcceptedKey, ConfigKey

from .preparerstep import PreparerStep


class IntentMapper(PreparerStep):
    """Apply intent resolution to each column.

    Params:
        *args: args to PreparerStep constructor.
        **kwargs: kwargs to PreparerStep constructor.

    """

    def __init__(self, **kwargs):
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
                IntentResolver(
                    column=column, cache_manager=self.cache_manager
                ),
                column,
            )
            for column in columns
        ]
        self.feature_processor = ColumnTransformerWrapper(
            list_of_tuples,
            n_jobs=self.cache_manager[AcceptedKey.CONFIG][ConfigKey.N_JOBS],
        )
        self.feature_processor.fit(X=X)
        self._update_cache_manager_with_intents()

        return self

    def transform(self, X, *args, **kwargs):
        """Transform X using this PreparerStep.

        calls underlying parallel process.

        Args:
            X: input DataFrame
            *args: args to .transform()
            **kwargs: kwargs to .transform()

        Returns:
            result from .transform()

        Raises:
            ValueError: if not fitted.

        """
        if self.feature_processor is None:
            raise ValueError("not fitted.")
        Xt = self.feature_processor.transform(X, *args, **kwargs)
        return Xt

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

    def _update_cache_manager_with_intents(self):
        for intent_resolver_tuple in self.feature_processor.transformers_:
            intent_resolver = intent_resolver_tuple[1]
            column_name = intent_resolver_tuple[2]
            self.cache_manager[AcceptedKey.INTENT][
                column_name
            ] = intent_resolver.column_intent
