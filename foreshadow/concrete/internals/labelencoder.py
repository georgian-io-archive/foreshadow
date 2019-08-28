"""FixedLabelEncoder."""

from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder

from foreshadow.base import BaseEstimator, TransformerMixin
from foreshadow.wrapper import pandas_wrap


@pandas_wrap
class FixedLabelEncoder(BaseEstimator, TransformerMixin):
    """Fix LabelEncoder function signature to fit transformer standard."""

    def __init__(self):
        self.encoder = SklearnLabelEncoder()

    def fit(self, X, y=None):
        """Fit the LabelEncoder.

        Args:
            X: iterable
            y (optional): iterable

        Returns:
            self

        """
        self.encoder.fit(X)
        return self

    def transform(self, X):
        """Transform using the fit LabelEncoder.

        Args:
            X: iterable

        Returns:
            array-like

        """
        return self.encoder.transform(X)

    def fit_transform(self, X):
        """Fit and transform in one step.

        Args:
            X: iterable

        Returns:
            array-like: Transformed samples

        """
        return self.encoder.fit_transform(X)

    def inverse_transform(self, X):
        """Transform labels back to original encoding.

        Args:
            X: iterable

        Returns:
            iterable: Inverted transformed samples

        """
        return self.encoder.inverse_transform(X)

    def get_params(self, deep=True):
        """Get parameters for this transformer. See super.

        Args:
            deep: deep to super get_params

        Returns:
            Params for this transformer. See super.

        """
        params = super().get_params(deep=deep)
        if not deep:
            params["encoder"] = self.encoder
        else:
            params["encoder"] = self.encoder.get_params(deep=deep)
        return params

    def set_params(self, **params):
        """Set parameters for this transformer. See super.

        Args:
            **params: params to set on this transformer.

        """
        self.encoder = params.pop("encoder")
        super().set_params(**params)
