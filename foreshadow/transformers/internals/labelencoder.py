from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder


class FixedLabelEncoder(BaseEstimator, TransformerMixin):
    """Fixes LabelEncoder function signature to fit transformer standard."""

    def __init__(self):
        self.encoder = SklearnLabelEncoder()

    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self

    def transform(self, X):
        return self.encoder.transform(X)

    def fit_transform(self, X):
        return self.encoder.fit_transform(X)

    def inverse_transform(self, X):
        return self.encoder.inverse_transform(X)
