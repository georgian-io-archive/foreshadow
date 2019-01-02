import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.base import TransformerMixin, BaseEstimator


class UncommonRemover(BaseEstimator, TransformerMixin):
    """Merges uncommon values in a categorical column to an other value

        Note: Unseen values from fitting will alse be merged.

        Args:
            threshold (float): data that is less frequant than this percentage will 
                be merged into a singular unique value
            replacement (Optional): value with which to replace uncommon values
    """

    def __init__(self, threshold=0.01, replacement=None):
        self.threshold = threshold
        self.replacement = replacement

    def fit(self, X, y=None):
        """
        Finds the uncommon values and sets the replacement value
        """
        X = check_array(X, dtype=None)
        self.values_, counts = np.unique(X, return_counts=True)
        self.merge_values_ = self.values_[np.where(counts <= self.threshold * X.size)]
        if self.replacement is None:
            if np.issubdtype(X.dtype, np.str_) or np.issubdtype(X.dtype, np.object_):
                self.replacement = "".join([str(i) for i in self.merge_values_])
            else:
                self.replacement = np.mean(self.merge_values_)

        return self

    def transform(self, X, y=None):
        X = check_array(X, dtype=None, copy=True)
        check_is_fitted(self, ["values_", "merge_values_"])
        np.putmask(
            X,
            np.isin(X, self.merge_values_) | ~np.isin(X, self.values_),
            self.replacement,
        )

        return X
