"""Provide functionality for PreparerSteps that need auto intent resolution."""

from .mapper import IntentMapper


class IntentResolverMixin:
    """Used for steps that need to resolve intents if not yet resolved."""

    def check_resolve(self, X, y=None, **fit_params):
        """Check if intents have been resolved and resolve if not.

        Args:
            X: input DataFrame to check
            y: labels
            **fit_params: params to fit

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
        columns_to_resolve = []
        for column in X.columns[0]:
            if self.column_sharer["intent", column] is None:
                columns_to_resolve.append(column)
        mapper = IntentMapper(column_sharer=self.column_sharer)
        X = mapper.fit_transform(X[columns_to_resolve], y=y, **fit_params)
