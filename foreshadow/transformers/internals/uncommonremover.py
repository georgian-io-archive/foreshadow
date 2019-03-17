from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from foreshadow.utils import check_df


class UncommonRemover(BaseEstimator, TransformerMixin):
    """Merges uncommon values in a categorical column to an other value

        Note: Unseen values from fitting will alse be merged.

        Args:
            threshold (float): data that is less frequant than this percentage
                will be merged into a singular unique value
            replacement (Optional): value with which to replace uncommon values
    """

    def __init__(self, threshold=0.01, replacement="UncommonRemover_Other"):
        self.threshold = threshold
        self.replacement = replacement

    def fit(self, X, y=None):
        """
        Finds the uncommon values and sets the replacement value

            Args:
                X (:obj:`pandas.DataFrame`): input dataframe
            returns:
                (self) object instance
        """
        X = check_df(X, single_column=True).iloc[:, 0]

        vc_series = X.value_counts()
        self.values_ = vc_series.index.values.tolist()
        self.merge_values_ = vc_series[
            vc_series <= (self.threshold * X.size)
        ].index.values.tolist()

        return self

    def transform(self, X, y=None):
        X = check_df(X, single_column=True).iloc[:, 0]
        check_is_fitted(self, ["values_", "merge_values_"])
        X[
            X.isin(self.merge_values_) | ~X.isin(self.values_)
        ] = self.replacement
        X = X.to_frame()

        return X
