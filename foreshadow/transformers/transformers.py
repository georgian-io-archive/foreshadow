"""Transformer wrapping utility classes and functions."""

import inspect
from functools import wraps

import numpy as np
import pandas as pd
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import VectorizerMixin

from foreshadow.utils import check_df


def is_transformer(cls):
    if issubclass(cls, BaseEstimator) and (
            issubclass(cls, TransformerMixin) or
            issubclass(cls, VectorizerMixin)
    ):
        return True
    return False


def _get_modules(classes, globals_, mname, wrap=True):
    """Import sklearn transformers from transformers directory.

    Searches transformers directory for classes implementing BaseEstimator and
    TransformerMixin and duplicates them, wraps their init methods and public
    functions to support pandas dataframes, and exposes them as
    foreshadow.transformers.[name]

    Returns:
        The list of wrapped transformers.

    """
    transformers = []

    for cls in classes:
        if is_transformer(cls):
            transformers.append(cls)

    if wrap:  # wrap the transformer with the
        transformer_wrapper = make_pandas_transformer
    else:
        transformer_wrapper = lambda x: x

    for t in transformers:
        copied_t = type(t.__name__, (t, *t.__bases__), dict(t.__dict__))
        copied_t.__module__ = mname
        globals_[copied_t.__name__] = transformer_wrapper(copied_t)

    return [t.__name__ for t in transformers]


def make_pandas_transformer(transformer):
    """Wrap an sklearn transformer to support dataframes.

    Args:
        transformer: sklearn transformer implementing BaseEstimator and
        TransformerMixin

    Returns:
        A wrapped form of transformer

    """
    declared_on = ['fit_transform']
    exists_on = ['fit', 'transform', 'inverse_transform']
    wrap_candidates = []
    for m in transformer.__dict__.values():
        if inspect.isfunction(m) and any(n == m.__name__ for n in declared_on):
            wrap_candidates.append(m)

    # normally, fit_transform will only be implemented in TransformerMixin
    # and will not need to be wrapped. Only if it is on the transformer
    # itself do we wrap it.
    members = [m[1]
               for m in inspect.getmembers(transformer,
                                           predicate=inspect.isfunction)]
    for m in members:
        if any(name == m.__name__ for name in exists_on):
            wrap_candidates.append(m)

    for w in wrap_candidates:
        # Wrap public function calls
        method = pandas_partial(w)
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

    """

    def __init__(self, src_func):
        """Save necessary info to copy over."""
        self.argspec = inspect.getfullargspec(src_func)
        self.src_doc = src_func.__doc__
        self.src_defaults = src_func.__defaults__

    def __call__(self, tgt_func):
        """Run when Sigcopy object is called. Returns new function."""
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
    """Partial function for injecting custom args into transformers."""

    def transform_constructor(
        self, *args, keep_columns=False, name=None, **kwargs
    ):

        self.name = args[-1]
        self.keep_columns = args[-2]
        func(self, *args[:-2], **kwargs)

    return transform_constructor


def pandas_partial(func):  # noqa: D202
    """Partial function for the pandas transformer wrapper."""

    @wraps(func)
    def pandas_func(self, *args, **kwargs):
        return pandas_wrapper(self, func, *args, **kwargs)

    return pandas_func


def init_replace(self, keep_columns=False, name=None):
    """Set the default values of custom transformer attributes."""
    self.keep_columns = keep_columns
    self.name = name


class _Empty(BaseEstimator, TransformerMixin):
    """Transformer that performs _Empty transformation."""

    def fit(self, X, y=None):
        """Empty fit function.

        Args:
            X (:obj:`numpy.ndarray`): Fit data

        Returns:
            self

        """
        return self

    def transform(self, X, y=None):
        """Pass through transform.

        Args:
            X (:obj:`numpy.ndarray`): X data

        Returns:
            :obj:`numpy.ndarray`: Empty numpy array

        """
        return pd.DataFrame([])

    def inverse_transform(self, X):
        """Pass through transform.

        Args:
            X (:obj:`numpy.ndarray`): X data

        Returns:
            :obj:`numpy.ndarray`: Empty numpy array

        """
        return pd.DataFrame([])


def _keep_columns_process(out, dataframe, prefix):
    """Keep original columns of input datafarme on output dataframe.

    Args:
        out: the output dataframe from the sklearn public function
        dataframe: the input dataframe from the sklearn public function
        prefix: the prefixes (name) to add

    Returns:
        [dataframe, out] concat along axis=1

    """
    dataframe.columns = [
        "{}_{}_origin_{}".format(c, prefix, i)
        for i, c in enumerate(dataframe.columns)
    ]
    return pd.concat([dataframe, out], axis=1)


def _ndarray_post_process(ndarray, index, init_cols, prefix):
    """Create dataframe from sklearn public function ndarray.

    Args:
        ndarray: the output ndarray from the sklearn public function
        init_cols: the initial columns before public function call
        prefix: prefix for each column (unique name)

    Returns:
        mimicked DataFrame for ndarray, with column names.

    """
    if ndarray.ndim == 1 and ndarray.size != 0:
        ndarray = ndarray.reshape((-1, 1))

    # Append new columns to data frame
    kw = {}
    for i, col in enumerate(ndarray.transpose().tolist()):
        kw["{}_{}_{}".format("_".join(init_cols), prefix, i)] = pd.Series(
                col, index=index
            )

    return pd.DataFrame(kw)


def _df_post_process(dataframe, init_cols, prefix):
    """Rename columns of output dataframe from sklearn public function.

    Args:
        dataframe: output DataFrame from sklearn public function
        init_cols: the initial columns before public function call
        prefix: prefix for each column (unique name)

    Returns:
        DataFrame with new column names

    """
    dataframe.columns = [
        "{}_{}_{}".format("_".join(init_cols), prefix, c)
        for c in dataframe.columns
    ]
    return dataframe


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
    if df.empty and not isinstance(self, _Empty):
        # this situation may happen when a transformer comes after the Empty
        # transformer in a pipeline. Sklearn transformers will break on empty
        # input and so we reroute to _Empty.
        func = getattr(_Empty, func.__name__)
    try:
        out = func(self, df, *args, **kwargs)
    except (TypeError, AttributeError):
        try:
            out = func(self, df, *args)
        except (AttributeError, ValueError):
            from sklearn.utils import check_array
            dat = check_array(
                df, accept_sparse=True, dtype=None, force_all_finite=False
            ).flatten()
            out = func(self, dat, *args)

    # determine name of new columns
    name = self.name if self.name else type(self).__name__
    out_is_transformer = hasattr(out, '__class__') and \
                         is_transformer(out.__class__)  # check if the output
    # returned by the sklearn public function is a transformer or not. It will
    # be a transformer in fit calls.

    if not (out_is_transformer):  #
        # if the output is a transformer, we do nothing.
        if isinstance(out, pd.DataFrame):  # custom handling based on the
            # type returned by the sklearn transformer function call
            out = _df_post_process(out, init_cols, name)
        elif isinstance(out, np.ndarray):
            out = _ndarray_post_process(out, df.index, init_cols, name)
        elif scipy.sparse.issparse(out):
            out = out.toarray()
            out = _ndarray_post_process(out, df, init_cols, name)
        elif isinstance(out, pd.Series):
            pass  # just return the series
        else:
            raise ValueError('undefined input {0}'.format(type(out)))

        if self.keep_columns:
            out = _keep_columns_process(out, df, name)
    return out
