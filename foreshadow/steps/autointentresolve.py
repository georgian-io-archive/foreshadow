"""Provide functionality for PreparerSteps that need auto intent resolution."""

from .mapper import IntentMapper


class AutoIntentMixin:
    """Used for steps that need to resolve intents if not yet resolved."""

    def check_resolve(self, X):
        """Check if intents have been resolved and resolve if not.

        Args:
            X: input DataFrame to check

        Raises:
            RuntimeError: If there is no self.column_sharer.

        """
        if getattr(self, "column_sharer", None) is None:
            raise RuntimeError(
                "Column Sharer was somehow None. Please make "
                "sure your class has a column_sharer "
                "attribute and that it extends from "
                "PreparerStep."
            )
        columns_to_resolve = [
            column
            for column in X.columns
            if self.column_sharer["intent", column] is None
        ]
        mapper = IntentMapper(column_sharer=self.column_sharer)
        X = mapper.fit_transform(X[columns_to_resolve])
