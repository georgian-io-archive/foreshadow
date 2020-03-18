"""Wrapper of truncated SVD for sparse matrices in Sklearn."""
from sklearn.decomposition import TruncatedSVD

from foreshadow.logging import logging


class TruncatedSVDWrapper(TruncatedSVD):
    """A wrapper of the Sklearn TruncatedSVD class."""

    def fit_transform(self, X, y=None):
        """Fit LSI model to X and perform dimensionality reduction on X.

        If the number of components specified is greater than the number of
        features, set the number of components to the number of features as a
        fall back.

        Args:
            X: {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data  # noqa
            y : Ignored

        Returns:
            X_new : array, shape (n_samples, n_components)
                Reduced version of X. This will always be a dense array.

        """
        n_features = X.shape[1]
        if self.n_components > n_features:
            logging.warning(
                "The number of components {} must be smaller than "
                "the number of features {} in the matrix. "
                "Fall back to {} components.".format(
                    self.n_components, n_features, n_features - 1
                )
            )
            self.n_components = n_features - 1
        res = super().fit_transform(X=X, y=y)
        return res
