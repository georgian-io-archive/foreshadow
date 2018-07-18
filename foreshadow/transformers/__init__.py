from copy import deepcopy
import glob
import inspect
import os

import numpy as np
import pandas as pd

from functools import partialmethod
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import (
    FeatureUnion,
    _fit_one_transformer,
    _fit_transform_one,
    _transform_one,
)
from sklearn.externals.joblib import Parallel, delayed, Memory


def get_classes():
    """Recursively searches transforms directory for classes"""

    files = glob.glob(os.path.dirname(__file__) + "/*.py")
    imports = [
        os.path.basename(f)[:-3]
        for f in files
        if os.path.isfile(f) and not f.endswith("__init__.py")
    ]
    modules = [__import__(i, globals(), locals(), ["object"], 1) for i in imports]
    classes = [
        c[1] for m in modules for c in inspect.getmembers(m) if inspect.isclass(c[1])
    ]

    return classes


def get_modules():
    """Recursively searches transforms directory for files containing classes
       extending sklearn.base.TransformerMixin
    """

    transformers = [
        cls
        for cls in get_classes()
        if issubclass(cls, TransformerMixin) and issubclass(cls, BaseEstimator)
    ]

    for t in transformers:
        copied_t = type(t.__name__, t.__bases__, dict(t.__dict__))
        globals()[copied_t.__name__] = wrap_transformer(copied_t)

    return len(transformers)


def wrap_transformer(transformer):
    """Wraps an sklearn transformer in functions allowing it to operate on a column basis
    """

    members = [
        m[1] for m in inspect.getmembers(transformer, predicate=inspect.isfunction)
    ]
    wrap_candidates = [
        m
        for m in members
        if "X" in inspect.getfullargspec(m).args and not m.__name__[0] == "_"
    ]

    for w in wrap_candidates:

        # Wrap public function calls
        setattr(transformer, w.__name__, partialmethod(pandas_wrapper, w))

    # Wrap constructor
    setattr(
        transformer,
        "__init__",
        partialmethod(transform_constructor, transformer.__init__),
    )

    return transformer


def transform_constructor(self, func, keep_columns=False, name=None, **kwargs):

    self.name = name
    self.keep_columns = keep_columns
    func(self, **kwargs)


def pandas_wrapper(self, func, df, *args, **kwargs):
    """Wrapper function to replace public transformer functions.
    Selects columns from df and executes inner function only on columns.

    This expects that public functions within the sklearn transformer follow the sklearn
    standard. This includes the format func(X, y=None, *args, **kwards) and either a
    return self or return X

    :arg func: Original function to be wrapped
    :arg self: Transformer instance object
    :arg df: Input data (pd.Dataframe)

    :return output: Returns transformer object or pd.Dataframe
    """

    stack = inspect.stack()
    caller = None
    try:
        caller = stack[1][0].f_locals["self"].__class__
        if caller.__name__ == type(self).__name__:
            return func(self, df, *args, **kwargs)
    except:
        pass

    init_cols = [col for col in df]
    out = func(self, df, *args, **kwargs)

    # If output is numpy array (transform has occurred)
    if isinstance(out, np.ndarray):

        # Remove old columns if necessary
        if not self.keep_columns:
            df = df[[]]

        # Determine name of new columns
        if self.name:
            prefix = self.name
        else:
            prefix = type(self).__name__

        # Append new columns to data frame
        for i, col in enumerate(out.transpose().tolist()):
            kw = {"{}_{}_{}".format("_".join(init_cols), prefix, i): pd.Series(col)}
            df = df.assign(**kw)

        return df

    return out


class ParallelProcessor(FeatureUnion):
    def __init__(self, transformer_list, n_jobs=1, transformer_weights=None):

        for i, item in enumerate(transformer_list):
            try:
                transformer_list[i][2].name = transformer_list[i][0]
            except AttributeError as e:
                pass

        super(ParallelProcessor, self).__init__(
            transformer_list, n_jobs, transformer_weights
        )

    def _update_transformer_list(self, transformers):
        transformers = iter(transformers)
        self.transformer_list[:] = [
            (name, cols, None if old is None else next(transformers))
            for name, cols, old in self.transformer_list
        ]

    def _validate_transformers(self):
        names, cols, transformers = zip(*self.transformer_list)

        # validate names
        self._validate_names(names)

        # validate estimators
        for t in transformers:
            if t is None:
                continue
            if not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(
                t, "transform"
            ):
                raise TypeError(
                    "All estimators should implement fit and "
                    "transform. '%s' (type %s) doesn't" % (t, type(t))
                )

    def _iter(self):
        """Generate (name, est, weight) tuples excluding None transformers
        """
        get_weight = (self.transformer_weights or {}).get

        return (
            (name, cols, trans, get_weight(name))
            for name, cols, trans in self.transformer_list
            if trans is not None
        )

    def fit(self, X, y=None):
        self.transformer_list = list(self.transformer_list)
        self._validate_transformers()

        transformers = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_transformer)(trans, X[cols], y)
            for name, cols, trans, weight in self._iter()
        )

        self._update_transformer_list(transformers)

        return self

    def transform(self, X):

        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, weight, X[cols])
            for name, cols, trans, weight in self._iter()
        )

        if not Xs:
            # All transformers are None
            return X[[]]
        else:
            Xs = pd.concat(Xs, axis=1)
        return Xs

    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()

        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, weight, X[cols], y, **fit_params)
            for name, cols, trans, weight in self._iter()
        )

        if not result:
            # All transformers are None
            return X[[]]
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)

        Xs = pd.concat(Xs, axis=1)
        return Xs


class SmartTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.transformer = None
        super(SmartTransformer, self).__init__()

    def _get_transformer(self, X, **fit_params):
        raise NotImplementedError(
            "WrappedTransformer _get_transformer was not implimented."
        )

    def _verify_transformer(self, X, **fit_params):

        if self.transformer:
            return

        self.transformer = self._get_transformer(X, **fit_params)

        if not self.transformer:
            raise AttributeError(
                "Invalid WrappedTransformer. Get transformer returns invalid object"
            )

        tf = getattr(self.transformer, "transform", None)
        fittf = getattr(self.transformer, "fit_transform", None)
        fit = getattr(self.transformer, "fit", None)

        if not (callable(tf) and callable(fittf) and callable(fit)):
            raise AttributeError(
                "Invalid WrappedTransformer. Get transformer returns invalid object"
            )

    def fit_transform(self, X, y=None, **fit_params):
        self._verify_transformer(X, **fit_params)
        return self.transformer.fit_transform(X, y, **fit_params)

    def transform(self, X, **kwargs):
        return self.transformer.transform(X, **kwargs)

    def fit(self, X, y=None, **kwargs):
        self._verify_transformer(X)
        return self.transformer.fit(X, y, **kwargs)


# TODO: Determine verbosity of this process. Global?
print("Loading transformers....")
n = get_modules()
print("Loaded {} transformer plugins".format(n))
