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
        return self

    def transform(self, X, *args, **kwargs):
        """Clean the dataframe.

        Args:
            X: the data frame.
            *args: positional args.
            **kwargs: key word args.

        Returns:
            A transformed dataframe.

        """
        summary = self.summarize(X)
        json.dump(summary, open("X_train_summary_upgrade.json", "w"), indent=4)
        return X

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

    # def fit_transform(self, X, y=None, **fit_params):
    #     """Fit then transform this PreparerStep.
    #
    #     Side-affect: Summarize each column.
    #
    #     Args:
    #         X: input DataFrame
    #         y: input labels
    #         **fit_params: kwarg params to fit
    #
    #     Returns:
    #         Result from .transform(), pass through.
    #
    #     """
    #     Xt = super().fit_transform(X, y, **fit_params)
    #     summary = self.summarize(Xt)
    #     json.dump(summary, open("X_train_summary.json", "w"), indent=4)
    #     return Xt

    def summarize(self, X_df):  # noqa
        summary = {}
        for k in X_df.columns.values.tolist():
            intent = self.cache_manager["intent", k]
            data = get_transformer(intent).column_summary(X_df[[k]])
            summary[k] = {"intent": intent, "data": data}
        return summary
