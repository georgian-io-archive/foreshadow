from sklearn.base import BaseEstimator, TransformerMixin


class FancyImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method="SimpleFill", **kwargs):
        self.kwargs = kwargs
        self.method = method
        try:
            module = __import__("fancyimpute", [method], 1)
            self.cls = getattr(module, method)
        except Exception as e:
            raise ValueError(
                "Invalid method. Possible values are BiScaler, KNN, "
                "NuclearNormMinimization and SoftImpute"
            )

        self.imputer = self.cls(**kwargs)

    def get_params(self, deep=True):
        return {"method": self.method, **self.kwargs}

    def set_params(self, **params):
        method = params.pop("method", self.method)
        print(params)

        self.kwargs = params
        self.method = method
        try:
            module = __import__("fancyimpute", [method], 1)
            self.cls = getattr(module, method)
        except Exception as e:
            raise ValueError(
                "Invalid method. Possible values are BiScaler, KNN, "
                "NuclearNormMinimization and SoftImpute"
            )

        self.imputer = self.cls(**params)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.imputer.complete(X)
