"""Provide functionality for PreparerSteps that need auto intent resolution."""

from .mapper import IntentMapper


class AutoIntentMixin:
    """Used for steps that need to resolve intents if not yet resolved."""

    def check_resolve(self, X):
        """Check if intents have been resolved and resolve if not.

        Args:
            X: input DataFrame to check

        Raises:
            RuntimeError: If there is no self.cache_manager.

        """
        if getattr(self, "cache_manager", None) is None:
            raise RuntimeError(
                "cache_manager was somehow None. Please make "
                "sure your class has a cache_manager "
                "attribute and that it extends from "
                "PreparerStep."
            )
        columns_to_resolve = [
            column
            for column in X.columns
            if self.cache_manager["intent", column] is None
        ]
        if len(columns_to_resolve) == 0:
            return

        # TODO do we really need IntentMapper? Is it possible that all the
        #  columns here Numerical?
        mapper = IntentMapper(cache_manager=self.cache_manager)
        _ = mapper.fit(X[columns_to_resolve])
