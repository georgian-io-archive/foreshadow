import numpy as np

from .transformers import SmartTransformer, FancyImputer, Imputer

from sklearn.pipeline import Pipeline


class SimpleImputer(SmartTransformer):
    def __init__(self, threshold=0.1, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def _get_transformer(self, X, y=None, **fit_params):
        s = X.ix[:, 0]
        ratio = s.isnull().sum() / s.count()

        if 0 < ratio <= self.threshold:
            return _choose_simple(s.values)
        else:
            return Pipeline([("null", None)])


class MultiImputer(SmartTransformer):
    def _get_transformer(self, X, y=None, **fit_params):
        if X.isnull().values.any():
            return _choose_multi(X)
        else:
            return Pipeline([("null", None)])


def _choose_simple(X):
    X = X[~np.isnan(X)]

    # Uses modified z score method http://colingorrie.github.io/outlier-detection.html
    # Assumes data is has standard distribution
    threshold = 3.5

    med_y = np.median(X)
    mad_y = np.median([np.abs(y - med_y) for y in X])
    z_scor = [0.6745 * (y - med_y) / mad_y for y in X]

    z_bool = np.where(np.abs(z_scor) > threshold)[0].shape[0] / X.shape[0] > 0.05

    if z_bool:
        # Impute with median
        print("USING MEDIAN")
        return FancyImputer("SimpleFill", fill_method="median")

    # Impute using mean
    print("USING MEAN")
    return FancyImputer("SimpleFill", fill_method="mean")


def _choose_multi(X):
    # For now simply default to KNN multiple imputation (generic case)
    # The rest of them seem to have constraints and no published directly comparable
    # performance

    # Impute using KNN
    return FancyImputer("KNN", k=3)
