# noqa
import json

from foreshadow.smart import FeatureSummarizer
from foreshadow.utils import get_transformer

from .autointentmap import AutoIntentMixin
from .preparerstep import PreparerStep


class FeatureSummarizerMapper(PreparerStep, AutoIntentMixin):  # noqa
    def __init__(self, **kwargs):
        """Define the single step for FeatureSummarizer.

        Args:
            **kwargs: kwargs to PreparerStep initializer.

        """
        super().__init__(**kwargs)

    def get_mapping(self, X):  # noqa
        return self.separate_cols(
            transformers=[
                [FeatureSummarizer(cache_manager=self.cache_manager)]
                for c in X
            ],
            cols=X.columns,
        )

    def fit_transform(self, X, y=None, **fit_params):
        """Fit then transform this PreparerStep.

        Side-affect: Summarize each column.

        Args:
            X: input DataFrame
            y: input labels
            **fit_params: kwarg params to fit

        Returns:
            Result from .transform(), pass through.

        """
        Xt = super().fit_transform(X, y, **fit_params)
        summary = self.summarize(Xt)
        json.dump(summary, open("X_train_summary.json", "w"), indent=4)
        return Xt

    def summarize(self, X_df):  # noqa

        return {
            k: {
                "intent": self.cache_manager["intent", k],
                "data": get_transformer(
                    self.cache_manager["intent", k]
                ).column_summary(X_df[[k]]),
            }
            for k in X_df.columns.values.tolist()
        }
