# noqa
import json

import pandas as pd

from foreshadow.utils import AcceptedKey, get_transformer, standard_col_summary

from .preparerstep import PreparerStep


class FeatureSummarizerMapper(PreparerStep):  # noqa
    def __init__(self, y_var=False, problem_type=None, **kwargs):
        """Define the single step for FeatureSummarizer.

        Args:
            y_var: whether the summerizer will be applied to X or y
            problem_type: when y_var is True, indicate whether this
                          is a regression or classification problem
            **kwargs: kwargs to PreparerStep initializer.

        """
        super().__init__(**kwargs)
        self.y_var = y_var
        self.problem_type = problem_type

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
        summary = self._summarize(X)

        if not self.y_var:
            json.dump(summary, open("X_train_summary.json", "w"), indent=4)

        summary_frame = self._cache_data_summary(summary)
        self.cache_manager[AcceptedKey.SUMMARY] = summary_frame
        return self

    def transform(self, X, *args, **kwargs):
        """Pass through transform.

        Args:
            X (:obj:`numpy.ndarray`): X data
            *args: positional args.
            **kwargs: key word args.

        Returns:
            :obj:`numpy.ndarray`: X

        """
        return X

    def inverse_transform(self, X):
        """Pass through transform.

        Args:
            X (:obj:`numpy.ndarray`): X data

        Returns:
            :obj:`numpy.ndarray`: X

        """
        return X

    def _summarize(self, X_df):
        summary = {}
        if self.y_var:
            intent = "Label"
            data = standard_col_summary(X_df)
            summary[X_df.columns[0]] = {"intent": intent, "data": data}
        else:
            for k in X_df.columns.values.tolist():
                intent = self.cache_manager[AcceptedKey.INTENT, k]
                data = get_transformer(intent).column_summary(X_df[[k]])
                summary[k] = {"intent": intent, "data": data}
        return summary

    def _convert_top(self, tops):
        result = {}
        accumulated_frequency = 0
        for i, (value, count, frequency) in enumerate(tops):
            accumulated_frequency += frequency
            result["#%d_value" % (i + 1)] = "%s %3.2f%%" % (
                value,
                accumulated_frequency * 100,
            )
        return result

    def _cache_data_summary(self, summary):
        records = {}
        for key, value in summary.items():
            rec = {"intent": value["intent"]}
            rec.update(value["data"])
            tops = rec.pop("top10")
            rec.update(self._convert_top(tops))
            records[key] = rec
        result = pd.DataFrame(records)
        result.fillna(value={"invalid_pct": 0.0}, axis=0, inplace=True)
        result.fillna("", inplace=True)
        result.sort_values(by="intent", axis=1, inplace=True)
        return result
