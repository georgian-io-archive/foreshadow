"""Fancy imputation."""

from sklearn.base import BaseEstimator, TransformerMixin


class FancyImputer(BaseEstimator, TransformerMixin):
    """Wrapper for the fancy imputation methods.

    Uses the `FancyImpute <https://github.com/iskandr/fancyimpute>` python
    package.

    Args:
        method (str): String of function from FancyImpute to invoke when
            transforming

    """

    def __init__(self, method="SimpleFill", **kwargs):
        self.kwargs = kwargs
        self.method = method
        try:
            module = __import__("fancyimpute", [method], 1)
            self.cls = getattr(module, method)
        except Exception:
            raise ValueError(
                "Invalid method. Possible values are BiScaler, KNN, "
                "NuclearNormMinimization and SoftImpute"
            )

        self.imputer = self.cls(**kwargs)

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator
                and contained subobjects that are estimators.

        Returns:
            dict: Parameter names mapped to their values.

        """
        return {"method": self.method, **self.kwargs}

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with :meth:`get_params()`.

        Args:
            **params: params to set

        Raises:
            ValueError: If method is invalid

        """
        method = params.pop("method", self.method)

        self.kwargs = params
        self.method = method

        # Auto import and initialize fancyimpute class defined by method
        try:
            module = __import__("fancyimpute", [method], 1)
            self.cls = getattr(module, method)
        except Exception:
            raise ValueError(
                "Invalid method. Possible values are BiScaler, KNN, "
                "NuclearNormMinimization and SoftImpute"
            )

        self.imputer = self.cls(**params)

    def fit(self, X, y=None):
        """Empty function.

        No fit necessary for these.

        Args:
            X: input observations
            y: input labels

        Returns:
            self

        """
        return self

    def transform(self, X):
        """Execute fancyimpute transformer on X data.

        Args:
            X (:obj:`pandas.DataFrame`): Input data

        Returns:
            :obj:`pandas.DataFrame`: Output data

        """
        return self.imputer.complete(X)
