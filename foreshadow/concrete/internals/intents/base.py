"""Base Intent for all intent definitions."""

from sklearn.base import BaseEstimator, TransformerMixin


class BaseIntent(BaseEstimator, TransformerMixin):
    """Base for all intent definitions.
    For each intent subclass a class attribute called `confidence_computation`
    must be defined which is of the form::
       {
            metric_def: weight
       }
    """

    @classmethod
    def get_confidence(cls, X, y=None):
        """Determine the confidence for an intent match.
        Args:
            X: input DataFrame.
            y: response variable
        Returns:
            float: A confidence value bounded between 0.0 and 1.0
        """
        return sum(
            [
                metric_fn(X) * weight
                for metric_fn, weight in cls.confidence_computation.items()
            ]
        )
