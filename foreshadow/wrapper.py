"""Transformer wrapping utility classes and functions."""

import warnings

import numpy as np
import pandas as pd
import scipy
from sklearn.base import BaseEstimator
from sklearn.utils.fixes import signature

from foreshadow.utils import check_df, is_transformer

from foreshadow.logging import logging
from .serializers import ConcreteSerializerMixin


def make_pandas_transformer(transformer):  # noqa: C901
    """Wrap a scikit-learn transformer to support pandas DataFrames.

    Args:
        transformer: scikit-learn transformer implementing
            `BaseEstimator <sklearn.base.BaseEstimator> and
            `TransformerMixin <sklearn.base.TransformerMixin>`

    Returns:
        The wrapped form of a transformer

    ..#noqa: I401

    """
    # use the same base metaclass as the transformer, otherwise we will get
    # MRO metaclass issues in DFTransformer if we try to choose the base class
    # for our metaclass that is not the same one for the transformer we are
    # also extending.
    class DFTransformerMeta(type(transformer)):
        """Metaclass for DFTransformer to appear as parent Transformer."""

        def __new__(meta, name, bases, class_dict):
            class_ = super().__new__(meta, name, bases, class_dict)

            if name == "DFTransformer":
                # If directly overriding a transformer using magic imports or
                # using decorator, imitate the parent class and register as the
                # parent class.
                # TODO (@Chris): The list of magic methods that are mapped
                # might need to increase.
                name_ = transformer.__name__
                class_.__name__ = name_
                class_.__doc__ = transformer.__doc__
            else:
                # If not directly wrapped, make sure to register the name of
                # the actual class being wrapped
                name_ = name

            # Only serialize if directly inheriting from SerializerMixin
            # if SerializerMixin in bases:
            #     register_transformer(class_, name_)
            # Unfortunately, polluting globals is the only way to
            # allow the pickling of wrapped transformers
            class_._repr_val = (
                transformer.__module__ + "." + transformer.__name__
            )
            globals()[name_] = class_
            class_.__qualname__ = name_

            return class_

        def __repr__(self):
            return self._repr_val

    class DFTransformer(
        transformer, ConcreteSerializerMixin, metaclass=DFTransformerMeta
    ):
        """Wrapper to Enable parent transformer to handle DataFrames."""

        def __init__(self, *args, **kwargs):
            """Initialize parent Transformer.

            Args:
                *args: args to the parent constructor (shadowed transformer)
                keep_columns: True to keep the original columns, False to not
                name: name for new/created columns
                **kwargs: kwargs to the parent constructor

            ..#noqa: I102

            """
            if "name" in kwargs:
                self.name = kwargs.pop("name")
                logging.warning(
                    "name is a deprecated kwarg. Please remove "
                    "it from the kwargs and instead set it "
                    "after instantiation."
                )
            if "keep_columns" in kwargs:
                self.keep_column = kwargs.pop("keep_columns")
                logging.warning(
                    "keep_columns is a deprecated kwarg. Please "
                    "remove it from the kwargs and instead set "
                    "it after instantiation."
                )
            try:
                super(DFTransformer, self).__init__(*args, **kwargs)
            except TypeError as e:
                raise type(e)(
                    str(e) + ". Init for transformer: '{}' "
                    "called".format(transformer)
                )

            # TODO: remove this when _Empty is removed
            self.__empty_fit = False

            self.is_wrapped = True

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
                    estimator and contained sub-objects that are estimators.

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
            try:
                self_params = super().get_params(deep=deep)
            except RuntimeError:
                # TODO, Chris explain why we copy scikit-learn's internal
                # get_params.
                self_params = dict()  # the output
                init = getattr(
                    self.__init__, "deprecated_original", self.__init__
                )
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
            """Fit the estimator or transformer, pandas enabled.

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

            out = func(df, *args, **kwargs)
            # determine name of new columns
            name = getattr(self, "name", type(self).__name__)
            out_is_transformer = hasattr(out, "__class__") and is_transformer(
                out.__class__
            )
            # check if the
            # output returned by the sklearn public function is a
            # transformer or not. It will be a transformer in fit calls.

            if not (out_is_transformer):
                # if the output is a transformer, we do nothing.
                if isinstance(
                    out, pd.DataFrame
                ):  # custom handling based on the
                    # type returned by the sklearn transformer function call
                    out, graph = _df_post_process(out, init_cols, name)
                elif isinstance(out, np.ndarray):
                    out, graph = _ndarray_post_process(
                        out, df, init_cols, name
                    )
                elif scipy.sparse.issparse(out):
                    out = out.toarray()
                    out, graph = _ndarray_post_process(
                        out, df, init_cols, name
                    )
                elif isinstance(out, pd.Series):
                    graph = []  # just return the series
                else:
                    raise ValueError("undefined output {0}".format(type(out)))

                if getattr(self, "keep_columns", False):
                    out = _keep_columns_process(out, df, name, graph)
                if getattr(self, "column_sharer", None) is not None:  # only
                    # used when part of the Foreshadow flow.
                    for column in X:
                        self.column_sharer["graph", column] = graph
                else:
                    logging.debug(
                        "column sharer is not set for: " "{}".format(self)
                    )
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
                ValueError: If not a valid output type from transformer.

            """
            df = check_df(X)

            init_cols = [str(col) for col in df]
            func = super(DFTransformer, self).inverse_transform

            out = func(df, *args, **kwargs)

            # determine name of new columns
            name = getattr(self, "name", type(self).__name__)
            out_is_transformer = hasattr(out, "__class__") and is_transformer(
                out.__class__, method="issubclass"
            )  # noqa: E127
            # check if the output
            # returned by the scikit-learn public function is a transformer or
            # not. It will be a transformer in fit calls.

            if not (out_is_transformer):
                # if the output is a transformer, we do nothing.
                if isinstance(
                    out, pd.DataFrame
                ):  # custom handling based on the
                    # type returned by the sklearn transformer function call
                    out, graph = _df_post_process(out, init_cols, name)
                elif isinstance(out, np.ndarray):
                    out, graph = _ndarray_post_process(
                        out, df, init_cols, name
                    )
                elif scipy.sparse.issparse(out):
                    out = out.toarray()
                    out, graph = _ndarray_post_process(
                        out, df, init_cols, name
                    )
                elif isinstance(out, pd.Series):
                    graph = []  # just return the series
                else:
                    raise ValueError("undefined input {0}".format(type(out)))

                if getattr(self, "keep_columns", False):
                    out = _keep_columns_process(out, df, name, graph)
                if getattr(self, "column_sharer", None) is not None:  # only
                    # used when part of the Foreshadow flow.
                    for column in X:
                        self.column_sharer["graph", column] = graph
                else:
                    logging.debug(
                        "column sharer is not set for: " "{}".format(self)
                    )
            return out  # TODO output is a DataFrame, make it detect based
            # TODO on what is passed to fit and give that output.

        def fit_transform(self, X, *args, **kwargs):
            df = check_df(X)
            kwargs.pop("full_df", None)
            init_cols = [str(col) for col in df]
            func = super(DFTransformer, self).fit_transform
            out = func(df, *args, **kwargs)

            # determine name of new columns
            name = getattr(self, "name", type(self).__name__)
            out_is_transformer = hasattr(out, "__class__") and is_transformer(
                out.__class__, method="issubclass"
            )  # noqa: E127
            # check if the output returned by the scikit-learn public function
            # is a transformer or not. It will be a transformer in fit calls.

            if not (out_is_transformer) and not isinstance(out, pd.DataFrame):
                # out_is_transformer: if the output is a transformer,
                # we do nothing.
                # pd.DataFrame: fit_transform will likely be
                # passed to the TransformerMixin fit_transform, which just
                # calls .fit and .transform. Processing will be handled
                # there
                if isinstance(out, np.ndarray):  # output was not yet
                    # transformed to DataFrame
                    out, graph = _ndarray_post_process(
                        out, df, init_cols, name
                    )
                elif scipy.sparse.issparse(out):
                    out = out.toarray()
                    out, graph = _ndarray_post_process(
                        out, df, init_cols, name
                    )
                elif isinstance(out, pd.Series):
                    graph = []  # just return the series
                else:
                    raise ValueError("undefined input {0}".format(type(out)))
                if getattr(self, "keep_columns", False):
                    out = _keep_columns_process(out, df, name, graph)
                if getattr(self, "column_sharer", None) is not None:  # only
                    # used when part of the Foreshadow flow.
                    for column in X:
                        self.column_sharer["graph", column] = graph
                else:
                    logging.debug(
                        "column sharer is not set for: " "{}".format(self)
                    )
            return out

        def __repr__(self):
            return "DFTransformer: {}".format(self.__class__.__name__)

        def set_extra_params(self, name=None, keep_columns=False):
            setattr(self, "name", name)
            setattr(self, "keep_columns", keep_columns)

    return DFTransformer


def _keep_columns_process(out, dataframe, prefix, graph):
    """Keep original columns of input datafarme on output dataframe.

    Args:
        out: the output dataframe from the sklearn public function
        dataframe: the input dataframe from the sklearn public function
        prefix: the prefixes (name) to add
        graph: current list representing information to add to graph in
            ColumnSharer

    Returns:
        [dataframe, out] concat along axis=1

    """
    graph.extend(
        [
            "{}_{}_origin_{}".format(c, prefix, i)
            for i, c in enumerate(dataframe.columns)
        ]
    )
    return pd.concat([dataframe, out], axis=1)


def _ndarray_post_process(ndarray, df, init_cols, prefix):
    """Create dataframe from sklearn public function ndarray.

    Args:
        ndarray: the output ndarray from the sklearn public function
        df: pandas.DataFrame
        init_cols: the initial columns before public function call
        prefix: prefix for each column (unique name)

    Returns:
        mimicked DataFrame for ndarray, with column names, list of info to
            graph in ColumnSharer

    """
    if ndarray.ndim == 1 and ndarray.size != 0:
        ndarray = ndarray.reshape((-1, 1))

    if ndarray.size == 0:
        return pd.DataFrame([]), ["{}_{}".format("_".join(init_cols), prefix)]
    # try to intelligently name ndarray columns, based off initial df columns
    if len(df.columns) == ndarray.shape[1]:  # the number of columns
        # match, so we don't have to do anything
        columns = df.columns
    elif len(df.columns) == 1:  # all new columns came from 1 column
        columns = [
            str(df.columns[0]) + "_{}".format(i)
            for i in range(ndarray.shape[1])
        ]
    else:  # all new columns came from a mix of columns
        df_columns = "_".join(df.columns)
        columns = [
            df_columns + "|{}".format(i) for i in range(ndarray.shape[1])
        ]
    # Append new columns to data frame
    kw = {}
    for i, col in enumerate(ndarray.transpose().tolist()):
        kw[columns[i]] = pd.Series(
            col, index=df.index  # noqa: E126
        )  # noqa: E121
    graph = [
        "{}_{}_{}".format("_".join(init_cols), prefix, i)
        for i in range(ndarray.shape[1])
    ]

    return pd.DataFrame(kw, columns=columns), graph


def _df_post_process(dataframe, init_cols, prefix):
    """Rename columns of output dataframe from sklearn public function.

    Args:
        dataframe: output DataFrame from sklearn public function
        init_cols: the initial columns before public function call
        prefix: prefix for each column (unique name)

    Returns:
        DataFrame with new column names, list of info to
            graph in ColumnSharer

    """
    graph = [
        "{}_{}_{}".format("_".join(init_cols), prefix, c)
        for c in dataframe.columns
    ]
    return dataframe, graph
