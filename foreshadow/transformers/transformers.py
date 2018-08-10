import inspect
from functools import partialmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils import check_df


def _get_modules(classes, globals_, mname):
    """Imports sklearn transformers from transformers directory.

    Searches transformers directory for classes implementing BaseEstimator and
    TransformerMixin and duplicates them, wraps their init methods and public functions
    to support pandas dataframes, and exposes them as foreshadow.transformers.[name]

    Returns:
        The number of transformers wrapped.

    """

    transformers = [
        cls
        for cls in classes
        if issubclass(cls, TransformerMixin) and issubclass(cls, BaseEstimator)
    ]

    for t in transformers:
        copied_t = type(t.__name__, t.__bases__, dict(t.__dict__))
        copied_t.__module__ = mname
        globals_[copied_t.__name__] = wrap_transformer(copied_t)

    return [t.__name__ for t in transformers]


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
    the original function ``(which should be using *args, **kwds)``.  The argspec,
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
    standard. This includes the format ``func(X, y=None, *args, **kwargs)`` and either a
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
            kw = {
                "{}_{}_{}".format("_".join(init_cols), prefix, i): pd.Series(
                    col, index=df.index
                )
            }
            df = df.assign(**kw)

        return df

    return out
