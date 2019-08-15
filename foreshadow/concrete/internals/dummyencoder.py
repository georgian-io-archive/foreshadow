"""DummyEncoder transformer."""

import pandas as pd
from sklearn.utils.validation import check_is_fitted

from foreshadow.base import BaseEstimator, TransformerMixin
from foreshadow.wrapper import pandas_wrap


@pandas_wrap
class DummyEncoder(BaseEstimator, TransformerMixin):
    """Dummy encode delimited data within column of dataframe."""

    def __init__(self, delimeter=",", other_cutoff=0.1, other_name="other"):
        self.delimeter = delimeter
        self.other_cutoff = other_cutoff
        self.other_name = other_name

    def fit(self, X, y=None):
        """Determine dummy categories.

        Args:
            X (:obj:`numpy.ndarray`): Fit data
            y: input labels

        Returns:
            self

        """
        X = X.iloc[:, 0]
        X = X.str.get_dummies(sep=self.delimeter)
        self.other = (X.fillna(0).sum(axis=0) / X.count()) < self.other_cutoff

        self.categories = [c for c in list(X) if not self.other[c]]
        self.other = [c for c in list(X) if self.other[c]]
        if len(self.other) > 0:
            self.categories += [self.other_name]

        return self

    def transform(self, X, y=None):
        """Perform dummy encoding on data.

        Args:
            X (:obj:`numpy.ndarray`): X data
            y: input labels

        Returns:
            :obj:`numpy.ndarray`: Transformed data

        """
        check_is_fitted(self, ["categories"])

        kwargs = {
            k: X.applymap(
                _separate(k, self.delimeter, self.other, self.other_name)
            )
            .iloc[:, 0]
            .tolist()
            for k in self.categories
        }
        df = pd.DataFrame(kwargs)

        return df


def _separate(cat, delim, other, other_name):  # noqa: D202
    """Get wrapped separate categories helper function.

    Args:
        cat: TODO(Adithya)
        delim: TODO(Adithya)
        other: TODO(Adithya)
        other_name: TODO(Adithya)

    Returns:
        TODO(Adithya)

    """

    def sep(X):
        if cat == other_name:
            if set(other) & set(X.split(delim)):
                return 1
            return 0
        if cat in X.split(delim):
            return 1
        return 0

    return sep
