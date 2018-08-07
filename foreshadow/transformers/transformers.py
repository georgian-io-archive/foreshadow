import inspect

import numpy as np
import pandas as pd

from functools import partialmethod
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import (
    FeatureUnion,
    _fit_one_transformer,
    _fit_transform_one,
    _transform_one,
)
from sklearn.externals.joblib import Parallel, delayed
from ..utils import check_df


def _get_classes():
    """Returns list of classes found in transforms directory."""

    module = __import__("externals", globals(), locals(), ["object"], 1)
    classes = [c[1] for c in inspect.getmembers(module) if inspect.isclass(c[1])]

    return classes


def _get_modules():
    """Imports sklearn transformers from transformers directory.

    Searches transformers directory for classes implementing BaseEstimator and
    TransformerMixin and duplicates them, wraps their init methods and public functions
    to support pandas dataframes, and exposes them as foreshadow.transformers.[name]

    Returns:
        The number of transformers wrapped.

    """

    transformers = [
        cls
        for cls in _get_classes()
        if issubclass(cls, TransformerMixin) and issubclass(cls, BaseEstimator)
    ]

    for t in transformers:
        copied_t = type(t.__name__, t.__bases__, dict(t.__dict__))
        copied_t.__module__ = "foreshadow.transformers"
        globals()[copied_t.__name__] = wrap_transformer(copied_t)

    return len(transformers)


def wrap_transformer(transformer):
    """Wraps an sklearn transformer to support dataframes.

    Args:
        transformer: sklearn transformer implementing BaseEstimator and TransformerMixin

    Returns:
        A wrapped form of transformer
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
    if "__defaults__" in dir(transformer.__init__):
        setattr(
            transformer,
            "__init__",
            Sigcopy(transformer.__init__)(init_partial(transformer.__init__)),
        )
    else:
        setattr(transformer, "__init__", init_replace)

    return transformer


class Sigcopy(object):
    """
    copy_argspec is a signature modifying decorator.  Specifically, it copies
    the signature from `source_func` to the wrapper, and the wrapper will call
    the original function (which should be using *args, **kwds).  The argspec,
    docstring, and default values are copied from src_func, and __module__ and
    __dict__ from tgt_func.
    """

    def __init__(self, src_func):
        self.argspec = inspect.getfullargspec(src_func)
        self.src_doc = src_func.__doc__
        self.src_defaults = src_func.__defaults__

    def __call__(self, tgt_func):
        tgt_argspec = inspect.getfullargspec(tgt_func)

        name = tgt_func.__name__
        argspec = self.argspec
        if argspec[3] is not None:
            argspec = (
                argspec[0:3]
                + (
                    tuple(
                        [s if type(s).__name__ != "type" else None for s in argspec[3]]
                    ),
                )
                + argspec[4:]
            )

        newargspec = (
            (argspec[0] + tgt_argspec[0][1:],)
            + argspec[1:4]
            + (tgt_argspec[4], tgt_argspec[5])
            + argspec[6:]
        )
        sigcall = inspect.formatargspec(formatvalue=lambda val: "", *newargspec)[1:-1]
        signature = inspect.formatargspec(*newargspec)[1:-1]

        signature = signature.replace("*,", "")
        sigcall = sigcall.replace("*,", "")

        new_func = (
            "def _wrapper_(%(signature)s):\n"
            "    return %(tgt_func)s(%(sigcall)s)"
            % {"signature": signature, "tgt_func": "tgt_func", "sigcall": sigcall}
        )

        evaldict = {"tgt_func": tgt_func}
        exec(new_func, evaldict)
        wrapped = evaldict["_wrapper_"]
        wrapped.__name__ = name
        wrapped.__doc__ = self.src_doc
        wrapped.__module__ = tgt_func.__module__
        wrapped.__dict__ = tgt_func.__dict__
        return wrapped


def init_partial(func):
    def transform_constructor(self, *args, keep_columns=False, name=None, **kwargs):

        self.name = args[-1]
        self.keep_columns = args[-2]
        func(self, *args[:-2], **kwargs)

    return transform_constructor


def init_replace(self, keep_columns=False, name=None):
    self.keep_columns = keep_columns
    self.name = name


def pandas_wrapper(self, func, df, *args, **kwargs):
    """Wrapper function to replace public transformer functions.

    Selects columns from df and executes inner function only on columns.

    This expects that public functions within the sklearn transformer follow the sklearn
    standard. This includes the format func(X, y=None, *args, **kwargs) and either a
    return self or return X

    Adds ability of transformer to handle DataFrame input and output with persistent
    column names.

    Args:
        self: The sklearn transformer object
        func: The original public function to be wrapped
        df: Pandas dataframe as input

    Returns:
        Same as return type of func
    """

    stack = inspect.stack()
    caller = None
    try:
        caller = stack[1][0].f_locals["self"].__class__
        if caller.__name__ == type(self).__name__:
            return func(self, df, *args, **kwargs)
    except:
        pass

    df = check_df(df)

    init_cols = [str(col) for col in df]
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
    """Class extending FeatureUnion to support parallel operation on dataframes.

    This class functions similarly to a FeatureUnion except it divides a given
    pandas dataframe according to the transformer definition in the constructure
    and runs the defined partial dataframes on the given transformer. It then
    concatenates the outputs together.

    Internally the ParallelProcessor uses MultiIndex-ing to indentify the column
    of origin for transformer operations that result in multiple columns.

    The outer index or 'origin' index represents the column used to create a
    calculated column or represents the leftmost column of a series of columns used
    to create a calculated
    column.

    By default the output contains both Index's to support pipeline usage and
    tracking for the Preprocessor. This can be suppressed.

    Attributes:
        collapse_index: Boolean defining whether multi-index should be flattened
        default_transformer_list: Transformer list shaped similar to FeatureUnion
        n_jobs: See base class
        transformer_weights: See base class
        transformer_list: List of transformer objects in form
            [(name, [cols], TransformerObject), ...]


    """

    def __init__(
        self, transformer_list, n_jobs=1, transformer_weights=None, collapse_index=False
    ):
        """Initializes ParallelProcessor class.

        Also recursively sets names of internal transformers and pipelines to the
        names given to them by the transformer_list definition.

        This allows the name of a transformer to be accessed internally by the object
        itself without referencing the parent ParallelProcessor.

        """

        self.collapse_index = collapse_index
        self.default_transformer_list = None

        for i, item in enumerate(transformer_list):
            self._set_names(item)

        super(ParallelProcessor, self).__init__(
            transformer_list, n_jobs, transformer_weights
        )

    def get_params(self, deep=True):
        """See base class."""

        self.default_transformer_list = [(a, c) for a, b, c in self.transformer_list]
        return self._get_params("default_transformer_list", deep=deep)

    def set_params(self, **kwargs):
        """See base class."""

        self.default_transformer_list = [(a, c) for a, b, c in self.transformer_list]
        return self._set_params("default_transformer_list", **kwargs)

    def _set_names(self, item):
        if hasattr(item[-1], "name"):
            item[-1].name = item[0]
        if hasattr(item[-1], "steps"):
            for step in item[-1].steps:
                self._set_names(step)
        if hasattr(item[-1], "transformer_list"):
            for trans in item[-1].transformer_list:
                self._set_names(trans)

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
        get_weight = (self.transformer_weights or {}).get

        return (
            (name, cols, trans, get_weight(name))
            for name, cols, trans in self.transformer_list
            if trans is not None
        )

    def _get_other_cols(self, X):

        full = set(list(X))
        partial = set(
            list(
                _slice_cols(
                    X,
                    [c for _, cols, _, _ in self._iter() for c in cols],
                    drop_level=False,
                )
            )
        )

        return list(full - partial)

    def fit(self, X, y=None):
        """See base class."""
        self.transformer_list = list(self.transformer_list)
        self._validate_transformers()

        transformers = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_transformer)(trans, _slice_cols(X, cols), y)
            for name, cols, trans, weight in self._iter()
        )

        self._update_transformer_list(transformers)

        return self

    def transform(self, X):
        """See base class."""
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_pandas_transform_one)(trans, weight, _slice_cols(X, cols), cols)
            for name, cols, trans, weight in self._iter()
        )

        Xo = X[self._get_other_cols(X)]
        if len(list(Xo)) > 0:
            if type(list(Xo)[0]) != tuple:
                Xo.columns = [list(Xo), list(Xo)]

            Xs += (Xo,)

        if not Xs:
            # All transformers are None
            return X[[]]
        else:
            Xs = pd.concat(Xs, axis=1)

        if self.collapse_index:
            Xs.columns = Xs.columns.droplevel()
        return Xs

    def fit_transform(self, X, y=None, **fit_params):
        """See base class."""
        self._validate_transformers()

        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_pandas_fit_transform_one)(
                trans, weight, _slice_cols(X, cols), y, cols, **fit_params
            )
            for name, cols, trans, weight in self._iter()
        )

        if not result:
            # All transformers are None
            return X[[]]

        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)

        Xo = X[self._get_other_cols(X)]

        if len(list(Xo)) > 0:
            if type(list(Xo)[0]) != tuple:
                Xo.columns = [list(Xo), list(Xo)]

            Xs += (Xo,)
        Xs = pd.concat(Xs, axis=1)

        if self.collapse_index:
            Xs.columns = Xs.columns.droplevel()
        return Xs


class SmartTransformer(BaseEstimator, TransformerMixin):
    """Abstract class following sklearn transformer standard.

    This class contains the logic necessary to determine a single transformer or
    pipeline object that should act in its place.

    Once in a pipeline this class can be continuously re-fit in order to adapt to
    different data sets.

    Contains a function _get_tranformer that must be ooverriddenby an implementing
    class that returns an sklearn transformer object to be used.

    Used and implements itself identically to a transformer.

    Attributes:
        override: An sklearn transformer that can be optionally provided to override
            internal logic.

    """

    def __init__(self, override=None, name=None, keep_columns=False, **kwargs):
        """Initializes SmartTransformer Object"""
        self.kwargs = kwargs
        self.name = name
        self.keep_columns = keep_columns
        self.override = override
        self.transformer = None

    def get_params(self, deep=True):
        return {
            "override": self.override,
            "name": self.name,
            "keep_columns": self.keep_columns,
            **(
                self.transformer.get_params(deep=True)
                if self.transformer is not None
                else {}
            ),
        }

    def set_params(self, **params):
        self.name = params.pop("name", self.name)
        self.keep_columns = params.pop("keep_columns", self.keep_columns)

        self.override = params.pop("override", self.override)
        if self.override:
            ovr = globals()[self.override]
            self.transformer = ovr(**self.kwargs)

        if self.transformer is not None:
            self.transformer.set_params(**params)

    def _get_transformer(self, X, y=None, **fit_params):
        raise NotImplementedError(
            "WrappedTransformer _get_transformer was not implimented."
        )

    def _verify_transformer(self, X, y=None, refit=False, **fit_params):

        if refit:
            self.transformer = None

        if self.transformer is not None:
            return

        if self.override is not None:
            ovr = globals()[self.override]
            self.transformer = ovr(**self.kwargs)
        else:
            self.transformer = self._get_transformer(X, y, **fit_params)

        if not self.transformer:
            raise AttributeError(
                "Invalid WrappedTransformer. Get transformer returns invalid object"
            )

        tf = getattr(self.transformer, "transform", None)
        fittf = getattr(self.transformer, "fit_transform", None)
        fit = getattr(self.transformer, "fit", None)

        nm = hasattr(self.transformer, "name")
        keep = hasattr(self.transformer, "keep_columns")

        pipe = hasattr(self.transformer, "steps")
        parallel = hasattr(self.transformer, "transformer_list")

        print(tf, fittf, fit, nm, keep, pipe, parallel)

        if not (
            callable(tf)
            and callable(fittf)
            and callable(fit)
            and (nm and keep)
            or pipe
            or parallel
        ):
            raise AttributeError(
                "Invalid WrappedTransformer. Get transformer returns invalid object"
            )

        self.transformer.name = self.name
        self.transformer.keep_columns = self.keep_columns

    def transform(self, X, **kwargs):
        """See base class."""
        X = check_df(X)
        self._verify_transformer(X, **kwargs)
        return self.transformer.transform(X, **kwargs)

    def fit(self, X, y=None, **kwargs):
        """See base class."""
        X = check_df(X)
        y = check_df(y)
        self._verify_transformer(X, y, refit=True)
        return self.transformer.fit(X, y, **kwargs)


def _slice_cols(X, cols, drop_level=True):
    """Searches for columns in dataframe using multi-index. """
    origin = list(X)
    if len(origin) == 0:
        return X
    if len(cols) == 0:
        return X.drop(list(X), axis=1)
    if type(origin[0]) == tuple:
        origin, new = list(zip(*origin))
    else:
        return X[cols]

    def get(c, level):
        ret = X.xs(c, axis=1, level=level, drop_level=False)
        if drop_level:
            ret.columns = ret.columns.droplevel()
        return ret

    df = pd.concat(
        [
            get(c.replace("$", ""), "new") if c[0] == "$" else get(c, "origin")
            for c in cols
            if c in origin or c.replace("$", "") in new
        ],
        axis=1,
    )

    return df


def _pandas_transform_one(transformer, weight, X, cols):
    """Transforms dataframe using sklearn transformer then adds multi-index"""
    colname = sorted(cols)[0]
    res = _transform_one(transformer, weight, X)
    res.columns = [[colname] * len(list(res)), list(res)]
    res.columns = res.columns.rename(["origin", "new"])
    return res


def _pandas_fit_transform_one(transformer, weight, X, y, cols, **fit_params):
    """Fits pandas dataframe, executes transformation, then adds multi-index. """
    colname = sorted(cols)[0]
    res, t = _fit_transform_one(transformer, weight, X, y, **fit_params)
    res.columns = [[colname] * len(list(res)), list(res)]
    res.columns = res.columns.rename(["origin", "new"])
    return res, t


print("Loading transformers....")
n = _get_modules()
print("Loaded {} transformer plugins".format(n))
