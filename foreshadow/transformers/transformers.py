"""Transformer wrapping utility classes and functions."""

import warnings
from functools import partial

import numpy as np
import pandas as pd
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import VectorizerMixin
from sklearn.utils.fixes import signature

from foreshadow.utils import check_df


def is_transformer(cls):
    """Check if the class is a transformer class.

    Args:
        cls: class

    Returns:
        True if transformer, False if not.

    """
    if issubclass(cls, BaseEstimator) and (
        issubclass(cls, TransformerMixin) or issubclass(cls, VectorizerMixin)
    ):
        return True
    return False


def _get_modules(classes, globals_, mname, wrap=True):
    """Import sklearn transformers from transformers directory.

    Searches transformers directory for classes implementing BaseEstimator and
    TransformerMixin and duplicates them, wraps their init methods and public
    functions to support pandas dataframes, and exposes them as
    foreshadow.transformers.[name]

    Args:
        classes: A list of classes
        globals_: The globals in the callee's context
        mname: The module name
        wrap: True to wrap modules, False to not (manual decorator
            make_panads_transformer) will need to be applied.

    Returns:
        The list of wrapped transformers.

    """

    def no_wrap(f):
        """Don't wrap function f. Return unwrapped function pointer.

        Args:
            f: function point f.

        Returns:
            function pointer, f

        """
        return f

    transformers = []

    for cls in classes:
        if is_transformer(cls):
            transformers.append(cls)

    if wrap:  # wrap the transformer with the
        transformer_wrapper = make_pandas_transformer
    else:
        transformer_wrapper = no_wrap

    for t in transformers:
        copied_t = type(t.__name__, (t, *t.__bases__), dict(t.__dict__))
        copied_t.__module__ = mname
        globals_[copied_t.__name__] = transformer_wrapper(copied_t)

    return [t.__name__ for t in transformers]


def make_pandas_transformer(transformer):
    """Wrap an sklearn transformer to support dataframes.

    Args:
        transformer: sklearn transformer implementing
            `BaseEstimator <sklearn.base.BaseEstimator> and
            `TransformerMixin <sklearn.base.TransformerMixin>`

    Returns:
        The wrapped form of a transformer

    ..#noqa: I401

    """
    # use the same base
    # metaclass as the transformer, otherwise we will get MRO metaclass
    # issues in DFTransformer if we try to choose the base class for our
    # metaclass that is not the same one for the transformer we are also
    # extending.
    class DFTransformerMeta(type(transformer)):
        """Metaclass for DFTransformer to appear as parent Transformer."""

        def __new__(mcs, *args, **kwargs):
            class_ = super(DFTransformerMeta, mcs).__new__(
                mcs, *args, **kwargs
            )
            class_.__name__ = transformer.__name__
            class_.__doc__ = transformer.__doc__
            return class_

    class DFTransformer(transformer, metaclass=DFTransformerMeta):
        """Wrapper to Enable parent transformer to handle DataFrames."""

        def __init__(self, *args, keep_columns=False, name=None, **kwargs):
            """Intiializes parent Transformer.

            Args:
                *args: args to the parent constructor (shadowed transformer)
                keep_columns: True to keep the original columns, False to not
                name: name for new/created columns
                **kwargs: kwargs to the parent constructor

            ..#noqa: I102

            """
            self.keep_columns = keep_columns
            self.name = name
            super(DFTransformer, self).__init__(*args, **kwargs)

        def get_params(self, deep=True):
            """Override standard get_params to handle nonstandard init.

            BaseEstimator for sklearn gets and sets parameters based on the
            init statement for that class. Since this class is used to wrap
            a parent transformer (by OOP), we use the parent's init
            statement and then this DFTransformer's additional arguments.
            We must override of BaseEstimator will complain about our
            nonstandard usage.

            Args:
                deep (bool): If True, will return the parameters for this
                    estimator and contained subobjects that are estimators.

            Returns:
                Parameter names mapped to their values for parent +
                DFTransformer wrapper.

            """
            parent_params = BaseEstimator.get_params(transformer, deep=deep)
            # will contain any init arguments that are not variable keyword
            # arguments. By default, this means that any new transformer
            # cannot have variable keyword arguments in its init less the
            # transformer designer is okay with it not getting picked up here.
            # The transformer class passed will not contain the current values,
            # so we set them with the values on the object instance, below.
            self_params = dict()  # the output
            init = getattr(self.__init__, "deprecated_original", self.__init__)
            if init is object.__init__:
                return self_params
            # explicit constructor to introspect
            # introspect the constructor arguments to find the model
            # parameters to represent
            init_signature = signature(init)
            # Consider the constructor parameters excluding 'self'
            self_sig = [
                p
                for p in init_signature.parameters.values()
                if p.name != "self"
                and p.kind != p.VAR_KEYWORD
                and p.kind != p.VAR_POSITIONAL
            ]
            self_sig = sorted([p.name for p in self_sig])
            for key in self_sig + list(parent_params.keys()):
                warnings.simplefilter("always", DeprecationWarning)
                try:
                    with warnings.catch_warnings(record=True) as w:
                        value = getattr(self, key, None)
                    if len(w) and w[0].category == DeprecationWarning:
                        # if the parameter is deprecated, don't show it
                        continue
                finally:
                    warnings.filters.pop(0)

                # XXX: should we rather test if instance of estimator?
                if deep and hasattr(value, "get_params"):
                    deep_items = value.get_params().items()
                    self_params.update(
                        (key + "__" + k, val) for k, val in deep_items
                    )
                self_params[key] = value
            return self_params

        def fit(self, X, *args, **kwargs):
            """Fit the estimtor or transformer, pandas enabled.

            See transformer.

            Args:
                X: inputs
                *args: arguments to transformer
                **kwargs: keyword arguments to transformer

            Returns:
                self

            """
            df = check_df(X)

            func = super(DFTransformer, self).fit
            if df.empty and not isinstance(self, _Empty):
                # this situation may happen when a transformer comes after the
                # Empty transformer in a pipeline. Sklearn transformers will
                # break on empty input and so we reroute to _Empty.
                func = partial(_Empty.fit, self)
            out = func(df, *args, **kwargs)
            return out

        def transform(self, X, y=None, *args, **kwargs):
            """Transform inputs using fitted transformer. Pandas enabled.

            See transformer

            Args:
                X: inputs
                y: labels
                *args: arguments to transformer
                **kwargs: keyword arguments to transformer

            Returns:
                transformed inputs

            Raises:
                ValueError: if not a valid output type from transformer

            """
            df = check_df(X)

            init_cols = [str(col) for col in df]
            func = super(DFTransformer, self).transform
            if df.empty and not isinstance(self, _Empty):
                # this situation may happen when a transformer comes after the
                # Empty transformer in a pipeline. Sklearn transformers will
                # break on empty input and so we reroute to _Empty.
                func = partial(_Empty.transform, self)

            out = func(df, *args, **kwargs)

            # determine name of new columns
            name = self.name if self.name else type(self).__name__
            out_is_transformer = hasattr(out, "__class__") and is_transformer(
                out.__class__
            )  # noqa: E127
            # check if the
            # output returned by the sklearn public function is a
            # transformer or not. It will be a transformer in fit calls.

            if not (out_is_transformer):
                # if the output is a transformer, we do nothing.
                if isinstance(
                    out, pd.DataFrame
                ):  # custom handling based on the
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
                    raise ValueError("undefined output {0}".format(type(out)))

                if self.keep_columns:
                    out = _keep_columns_process(out, df, name)
            return out

        def inverse_transform(self, X, *args, **kwargs):
            """Give original inputs using fitted transformer. Pandas enabled.

            See transformer

            Args:
                X: transformed inputs
                *args: arguments to transformer
                **kwargs: keyword arguments to transformer

            Returns:
                original inputs

            Raises:
                ValueError: if not a valid output type from transformer

            """
            df = check_df(X)

            init_cols = [str(col) for col in df]
            func = super(DFTransformer, self).inverse_transform
            if df.empty and not isinstance(self, _Empty):
                # this situation may happen when a transformer comes after the
                # Empty transformer in a pipeline. Sklearn transformers will
                # break on empty input and so we reroute to _Empty.
                func = partial(_Empty.inverse_transform, self)

            out = func(df, *args, **kwargs)

            # determine name of new columns
            name = self.name if self.name else type(self).__name__
            out_is_transformer = hasattr(out, "__class__") and is_transformer(
                out.__class__
            )  # noqa: E127
            # check if the output
            # returned by the sklearn public function is a transformer or not.
            # It will be a transformer in fit calls.

            if not (out_is_transformer):
                # if the output is a transformer, we do nothing.
                if isinstance(
                    out, pd.DataFrame
                ):  # custom handling based on the
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
                    raise ValueError("undefined input {0}".format(type(out)))

                if self.keep_columns:
                    out = _keep_columns_process(out, df, name)
            return out

        def fit_transform(self, X, *args, **kwargs):
            df = check_df(X)
            kwargs.pop("full_df", None)
            init_cols = [str(col) for col in df]
            func = super(DFTransformer, self).fit_transform
            out = func(df, *args, **kwargs)

            # determine name of new columns
            name = self.name if self.name else type(self).__name__
            out_is_transformer = hasattr(out, "__class__") and is_transformer(
                out.__class__
            )  # noqa: E127
            # check if the output returned by the sklearn public function is
            # a transformer or not. It will be a transformer in fit calls.

            if not (out_is_transformer) and not isinstance(out, pd.DataFrame):
                # out_is_transformer: if the output is a transformer,
                # we do nothing.
                # pd.DataFrame: fit_transform will likely be
                # passed to the TransformerMixin fit_transform, which just
                # calls .fit and .transform. Processing will be handled
                # there
                if isinstance(out, np.ndarray):  # output was not yet
                    # transformed to DataFrame
                    out = _ndarray_post_process(out, df.index, init_cols, name)
                elif scipy.sparse.issparse(out):
                    out = out.toarray()
                    out = _ndarray_post_process(out, df, init_cols, name)
                elif isinstance(out, pd.Series):
                    pass  # just return the series
                else:
                    raise ValueError("undefined input {0}".format(type(out)))
                if self.keep_columns:
                    out = _keep_columns_process(out, df, name)
            return out

    return DFTransformer


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
        index: pandas.DataFrame.index
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
            col, index=index  # noqa: E126
        )  # noqa: E121

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
