"""Wrapper for any sklearn transformer to ensure it can handle pandas input.


Requirements and Concerns:
    1. Take outputted transformers and use them elsewhere
    2. Swap in other transformers and use them in the pipeline
        * Note, only possible if using the foreshadow versions.
    3. Wrap sklearn transformer public methods
        a. They output ndarrays
        b. They should be able to take in dataframes regardless
    4. Have transformer appear as standard sklearn transformer (isinstance)
    5.

Options:
    1. wrap class at import time
        a. overwrite methods
        --> less flexibility, each class has to be handled the same, less the
        code gets too unmaintainable with many if else's etc.
        b. overwrite class
        Generally, this will force needing foreshadow versions, but
        leave the pipeline untouched
        --> more flexibile, but will not appear as same type (no use of
        sigcopy).
    --> for either, we can just separate out the code for better style and
    we should be good
    2. write custom pipeline
        a. wrap fit calls and mirror internally
        Drastically reduces code for wrapping classes, but transformers are not
        DataFrame encoders anymore.


"""


import inspect
import scipy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.feature_extraction.text import VectorizerMixin
# from utils import check_df


def _get_classes():
    """Return a list of classes found in transforms directory."""
    return [c for c in globals().values() if inspect.isclass(c)]


def _get_modules(classes, globals_, mname):
    """Import sklearn transformers from transformers directory.

    Searches transformers directory for classes implementing BaseEstimator and
    TransformerMixin and duplicates them, wraps their init methods and public
    functions to support pandas dataframes, and exposes them as
    foreshadow.transformers.[name]

    Returns:
        The list of wrapped transformers.

    """
    transformers = [
        cls
        for cls in classes
        if issubclass(cls, BaseEstimator)
        and (
            issubclass(cls, TransformerMixin)
            or issubclass(cls, VectorizerMixin)
        )
    ]

    for t in transformers:
        copied_t = type(t.__name__, (t, *t.__bases__), dict(t.__dict__))
        copied_t.__module__ = mname
        globals_[copied_t.__name__] = wrap_transformer(copied_t)

    return [t.__name__ for t in transformers]


def wrap_transformer(transformer):
    """Wrap an sklearn transformer to support dataframes.

    Args:
        transformer: sklearn transformer implementing BaseEstimator and
        TransformerMixin

    Returns:
        A wrapped form of transformer

    """
    class DFTransformer(transformer):
        def fit(self):
            pass
        def transform(self):
            pass

    return DFTransformer


def pandas_wrapper_v2(self, func, df, *args, **kwargs):
    df = check_df(df)
    init_cols = [str(col) for col in df]

    func()


def pandas_wrapper(self, func, df, *args, **kwargs):
    """Replace public transformer functions using wrapper.

    Selects columns from df and executes inner function only on columns.

    This expects that public functions within the sklearn transformer follow
    the sklearn standard. This includes the format
    ``func(X, y=None, *args, **kwargs)`` and either a return self or return X

    Adds ability of transformer to handle DataFrame input and output with
    persistent column names.

    Args:
        self: The sklearn transformer object
        func: The original public function to be wrapped
        df: Pandas dataframe as input

    Returns:
        Same as return type of func

    """

    df = check_df(df)

    init_cols = [str(col) for col in df]
    if not df.empty or isinstance(self, _Empty):
        try:
            out = func(self, df, *args, **kwargs)
        except Exception:
            try:
                out = func(self, df, *args)
            except Exception:
                from sklearn.utils import check_array

                dat = check_array(
                    df, accept_sparse=True, dtype=None, force_all_finite=False
                ).tolist()
                dat = [i for t in dat for i in t]
                out = func(self, dat, *args)
    else:
        fname = func.__name__
        if "transform" in fname:
            out = df
        else:  # fit
            out = _Empty().fit(df)

    # If output is DataFrame (custom transform has occured)
    if isinstance(out, pd.DataFrame):
        if hasattr(out, "from_transformer"):
            return out

        if self.name:
            prefix = self.name
        else:
            prefix = type(self).__name__

        out.columns = [
            "{}_{}_{}".format("_".join(init_cols), prefix, c)
            for c in out.columns
        ]

        if self.keep_columns:
            df.columns = [
                "{}_{}_origin_{}".format(c, prefix, i)
                for i, c in enumerate(df.columns)
            ]
            return pd.concat([df, out], axis=1)

        out.from_transformer = True

        return out

    if scipy.sparse.issparse(out):  # densify sparse matricies
        out = out.toarray()

    # If output is numpy array (transform has occurred)
    if isinstance(out, np.ndarray):  # TODO can we abstract away this code,
        # TODO combine as much as possible.

        # Remove old columns if necessary
        if not self.keep_columns:
            df = df[[]]

        # Determine name of new columns
        if self.name:
            prefix = self.name
        else:
            prefix = type(self).__name__

        if out.ndim == 1 and out.size != 0:
            out = out.reshape((-1, 1))

        # Append new columns to data frame
        for i, col in enumerate(out.transpose().tolist()):
            kw = {
                "{}_{}_{}".format("_".join(init_cols), prefix, i): pd.Series(
                    col, index=df.index
                )
            }
            df = df.assign(**kw)

        df.from_transformer = True

        return df

    return out


classes = _get_modules(_get_classes(), globals(), __name__)


if __name__ == '__main__':
    relative_data_path = "../tests/test_data/"
    relative_config_path = "../tests/test_configs/"


    def foreshadow_param_optimize_fit():
        import pandas as pd
        from sklearn.base import BaseEstimator, TransformerMixin
        from sklearn.model_selection._search import BaseSearchCV

        from foreshadow import Foreshadow

        data = pd.read_csv(relative_data_path+"boston_housing.csv")

        class DummyRegressor(BaseEstimator, TransformerMixin):
            def fit(self, X, y):
                return self

        class DummySearch(BaseSearchCV):
            def __init__(self, estimator, params):
                self.best_estimator_ = estimator

            def fit(self, X, y=None, **fit_params):
                return self

        class DummyPreprocessor(BaseEstimator, TransformerMixin):
            def fit(self, X, y):
                return self

        # mock_p.return_value = DummyPreprocessor()

        fs = Foreshadow(estimator=DummyRegressor(), optimizer=DummySearch)
        x = data.drop(["medv"], axis=1, inplace=False)
        y = data[["medv"]]

        fs.fit(x, y)
        assert isinstance(fs.pipeline.steps[-1][1].estimator, DummyRegressor)

        fs2 = Foreshadow(
            X_preprocessor=False,
            y_preprocessor=False,
            estimator=DummyRegressor(),
            optimizer=DummySearch,
        )

        fs2.fit(x, y)
        assert isinstance(fs2.pipeline.steps[-1][1], DummyRegressor)


    def smart_text():
        import numpy as np
        import pandas as pd

        from foreshadow.transformers.smart import SmartText
        from foreshadow.transformers.externals import TfidfVectorizer
        from foreshadow.transformers.internals import HTMLRemover

        X1 = pd.DataFrame(["abc", "def", "1321", "tester"])
        tf1 = SmartText().fit(X1)

        assert isinstance(tf1, TfidfVectorizer)

        X2 = pd.DataFrame(["<p> Hello </p>", "World", "<h1> Tag </h1>"])
        tf2 = SmartText().fit(X2)
        assert any(isinstance(tf, HTMLRemover) for n, tf in tf2.steps)
        assert isinstance(tf2.steps[-1][1], TfidfVectorizer)

        assert SmartText().fit(pd.DataFrame([1, 2, 3, np.nan]))
    def transformer_wrapper_empty_input():
        import numpy as np
        import pandas as pd

        from sklearn.preprocessing import StandardScaler as StandardScaler
        from foreshadow.transformers.externals import (
            StandardScaler as CustomScaler,
        )

        df = pd.DataFrame({"A": np.array([])})

        # with pytest.raises(ValueError) as e:
        #     StandardScaler().fit(df)
        cs = CustomScaler().fit(df)
        out = cs.transform(df)

        # assert "Found array with" in str(e)
        assert out.values.size == 0

        # If for some weird reason transform is called before fit
        assert CustomScaler().transform(df).values.size == 0
    def smart_emtpy_input():
        import numpy as np

        from foreshadow.transformers.smart import Scaler
        from foreshadow.transformers.transformers import _Empty

        normal_data = np.array([])
        smart_scaler = Scaler()
        assert isinstance(smart_scaler.fit(normal_data), _Empty)
        assert smart_scaler.transform(normal_data).values.size == 0
    def preprocessor_fit_transform():
        import json
        import pandas as pd
        from foreshadow.preprocessor import Preprocessor

        df = pd.read_csv(relative_data_path+"boston_housing.csv")

        truth = pd.read_csv(relative_data_path+"boston_housing_processed.csv",
                            index_col=0
                            )
        config = json.load(open("complete_config.json", 'r'))
        proc = Preprocessor(
            from_json=config
        )
        proc.fit(df)
        out = proc.transform(df)
        print(set([c for l in list(out) for c in l.split("_")]))
        print(set(
            [c for l in list(truth) for c in l.split("_")]
        ))

        assert set([c for l in list(out) for c in l.split("_")]) == set(
            [c for l in list(truth) for c in l.split("_")]
        )


    # preprocessor_fit_transform()
    # from sklearn.decomposition import PCA
    from sklearn.decomposition.base import _BasePCA
    # PCA = make_pandas_transformer(PCA)
    # smart_emtpy_input()
    # foreshadow_param_optimize_fit()
    # smart_text()
    # transformer_wrapper_empty_input()
    print("done")