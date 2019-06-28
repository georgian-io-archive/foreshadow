"""Transformer wrapping utility classes and functions."""

import inspect
from functools import wraps

import numpy as np
import pandas as pd
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import VectorizerMixin

from foreshadow.utils import check_df


def _get_modules(classes, _globals, mname):
    """Import sklearn transformers from transformers directory.

    Searches transformers directory for classes implementing BaseEstimator and
    TransformerMixin and duplicates them, wraps their init methods and public
    functions to support pandas dataframes, and exposes them as
    foreshadow.transformers.[name]

    Args:
        classes: A list of classes
        _globals: The globals in the callee's context
        mname: The module name

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
        _globals[copied_t.__name__] = wrap_transformer(copied_t)

    return [t.__name__ for t in transformers]


def wrap_transformer(transformer):
    """Wrap an sklearn transformer to support dataframes.

    Args:
        transformer: sklearn transformer implementing
            `BaseEstimator <sklearn.base.BaseEstimator> and
            `TransformerMixin <sklearn.base.TransformerMixin>`

    Returns:
        The wrapped form of a transformer

    """
    members = [
        m[1]
        for m in inspect.getmembers(transformer, predicate=inspect.isfunction)
    ]
    wrap_candidates = [
        m
        for m in members
        if any(
            name == m.__name__
            for name in [
                "fit",
                "fit_transform",
                "transform",
                "inverse_transform",
            ]
        )
    ]

    for w in wrap_candidates:
        # Wrap public function calls
        method = pandas_partial(pandas_partial(w))
        method.__doc__ = w.__doc__
        setattr(transformer, w.__name__, method)

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
    """Copy the argspec between two functions.

    Used to copy the argspec from a partial function.

    Params:
        src_func: The source function to copy

    Call Params:
        tgt_func: The target function to emulate

    """

    def __init__(self, src_func):
        """Save necessary info to copy over.

        Args:
            src_func: The source function to be copied.

        """
        self.argspec = inspect.getfullargspec(src_func)
        self.src_doc = src_func.__doc__
        self.src_defaults = src_func.__defaults__

    def __call__(self, tgt_func):
        """Run when Sigcopy object is called. Returns new function.

        Args:
            tgt_func: The target function to emulate

        Returns:
            source function wrapped as target function.
            TODO(@Adithya) fix this docstring.

        """
        tgt_argspec = inspect.getfullargspec(tgt_func)

        name = tgt_func.__name__

        # Filters out defaults that are metaclasses
        argspec = self.argspec
        argspec = (
            argspec[0:3]
            + (
                tuple(
                    [
                        s if type(s).__name__ != "type" else None
                        for s in (argspec[3] if argspec[3] is not None else [])
                    ]
                ),
            )
            + argspec[4:]
        )

        # Copies keyword arguments and defaults
        newargspec = (
            (argspec[0] + tgt_argspec[0][1:],)
            + argspec[1:4]
            + (tgt_argspec[4], tgt_argspec[5])
            + argspec[6:]
        )

        # Write new function
        sigcall = inspect.formatargspec(
            formatvalue=lambda val: "", *newargspec
        )[1:-1]
        signature = inspect.formatargspec(*newargspec)[1:-1]

        signature = signature.replace("*,", "")
        sigcall = sigcall.replace("*,", "")

        new_func = (
            "def _wrapper_(%(signature)s):\n"
            "    return %(tgt_func)s(%(sigcall)s)"
            % {
                "signature": signature,
                "tgt_func": "tgt_func",
                "sigcall": sigcall,
            }
        )

        # Set new metadata
        evaldict = {"tgt_func": tgt_func}
        exec(new_func, evaldict)
        wrapped = evaldict["_wrapper_"]
        wrapped.__name__ = name
        wrapped.__doc__ = self.src_doc
        wrapped.__module__ = tgt_func.__module__
        wrapped.__dict__ = tgt_func.__dict__
        return wrapped


def init_partial(func):  # noqa: D202
    """Partial function for injecting custom args into transformers.

    Args:
        func: function to wrap

    Returns:
        wrapped function

    """

    def transform_constructor(
        self, *args, keep_columns=False, name=None, **kwargs
    ):

        self.name = args[-1]
        self.keep_columns = args[-2]
        func(self, *args[:-2], **kwargs)

    return transform_constructor


def pandas_partial(func):  # noqa: D202
    """Partial function for the pandas transformer wrapper.

    Args:
        func: function to wrap

    Returns:
        wrapped function

    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return pandas_wrapper(self, func, *args, **kwargs)

    return wrapper


def init_replace(transformer, keep_columns=False, name=None):
    """Set the default values of custom transformer attributes.

    Args:
        transformer: transformer object
        keep_columns: keep_columns value for transformer
        name: name value for transformer

    """
    transformer.keep_columns = keep_columns
    transformer.name = name


class _Empty(BaseEstimator, TransformerMixin):
    """Transformer that performs _Empty transformation."""

    def fit(self, X, y=None):
        """Empty fit function.

        Args:
            X (:obj:`numpy.ndarray`): input data to fit, observations
            y: labels

        Returns:
            self

        """
        return self

    def transform(self, X, y=None):
        """Pass through transform.

        Args:
            X (:obj:`numpy.ndarray`): X data
            y: labels

        Returns:
            :obj:`numpy.ndarray`: Empty numpy array

        """
        return np.array([])

    def inverse_transform(self, X):
        """Pass through transform.

        Args:
            X (:obj:`numpy.ndarray`): X data

        Returns:
            :obj:`numpy.ndarray`: Empty numpy array

        """
        return np.array([])


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
        *args: args for func
        **kwargs: kwargs for func

    Returns:
        Same as return type of func

    """
    current = inspect.currentframe()
    calframe = inspect.getouterframes(current, 3)
    if calframe[2][3] != "pandas_wrapper":
        return func(self, df, *args, **kwargs)

    df = check_df(df)

    init_cols = [str(col) for col in df]
    if not df.empty or isinstance(self, _Empty):
        try:
            out = func(self, df, *args, **kwargs)
        except Exception:
            out = func(self, df, *args)
    else:
        fname = func.__name__
        if "transform" in fname:
            out = df
        else:  # fit
            out = _Empty().fit(df)

    # If output is DataFrame (custom transform has occurred)
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
    if isinstance(out, np.ndarray):

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


# Wrap _Empty manually after function definition
_Empty = wrap_transformer(_Empty)
