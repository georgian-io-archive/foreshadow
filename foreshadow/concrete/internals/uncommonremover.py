"""Uncommon remover."""

from sklearn.utils.validation import check_is_fitted

from foreshadow.base import BaseEstimator, TransformerMixin
from foreshadow.utils import check_df
from foreshadow.wrapper import pandas_wrap


@pandas_wrap
class UncommonRemover(BaseEstimator, TransformerMixin):
    """Merge uncommon values in a categorical column to an other value.

    Note: Unseen values from fitting will also be merged.

    Args:
        threshold (float): data that is less frequent than this percentage
            will be merged into a singular unique value
        replacement (Optional): value with which to replace uncommon values

    """

    def __init__(self, threshold=0.01, replacement="UncommonRemover_Other"):
        self.threshold = threshold
        self.replacement = replacement

    def fit(self, X, y=None):
        """Find the uncommon values and set the replacement value.

        Args:
            X (:obj:`pandas.DataFrame`): input dataframe
            y: input labels

        Returns:
            self

        """
        X = check_df(X, single_column=True).iloc[:, 0]

        vc_series = X.value_counts()
        self.values_ = vc_series.index.values.tolist()
        self.merge_values_ = vc_series[
            vc_series <= (self.threshold * X.size)
        ].index.values.tolist()

        return self

    def transform(self, X, y=None):
        """Apply the computed transform to the passed in data.

        Args:
            X (:obj:`pandas.DataFrame`): input DataFrame
            y: input labels

        Returns:
            :obj:`pandas.DataFrame`: transformed dataframe

        """
        X = check_df(X, single_column=True).iloc[:, 0]
        check_is_fitted(self, ["values_", "merge_values_"])
        X[
            X.isin(self.merge_values_) | ~X.isin(self.values_)
        ] = self.replacement
        X = X.to_frame()

        return X
