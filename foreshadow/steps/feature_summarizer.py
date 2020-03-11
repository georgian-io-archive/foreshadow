# noqa
import json

from foreshadow.intents import Droppable
from foreshadow.utils import AcceptedKey, get_transformer

from .preparerstep import PreparerStep


class FeatureSummarizerMapper(PreparerStep):  # noqa
    def __init__(self, **kwargs):
        """Define the single step for FeatureSummarizer.

        Args:
            **kwargs: kwargs to PreparerStep initializer.

        """
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
        summary = self._summarize(X)
        json.dump(summary, open("X_train_summary.json", "w"), indent=4)
        return X

    def _summarize(self, X_df):
        summary = {}
        for k in X_df.columns.values.tolist():
            intent = self.cache_manager[AcceptedKey.INTENT, k]
            if intent != Droppable.__name__:
                data = get_transformer(intent).column_summary(X_df[[k]])
                summary[k] = {"intent": intent, "data": data}
        return summary
