"""Transformer wrapping utility classes and functions."""

import numpy as np
import pandas as pd
import scipy

from foreshadow.logging import logging
from foreshadow.utils import check_df, is_transformer


def pandas_wrap(transformer):  # noqa
    """Wrap a scikit-learn transformer to support pandas DataFrames.

    Args:
        transformer: scikit-learn transformer implementing
            `BaseEstimator <sklearn.base.BaseEstimator> and
            `TransformerMixin <sklearn.base.TransformerMixin>`

    Returns:
        The wrapped form of a transformer

    ..# noqa: I401
    ..# noqa: DAR401

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

    class DFTransformer(transformer, metaclass=DFTransformerMeta):
        """Wrapper to Enable parent transformer to handle DataFrames."""

        def __init__(self, *args, name=None, keep_columns=False, **kwargs):
            # noqa
            """Initialize parent Transformer.

            Args:
                *args: args to the parent constructor (shadowed transformer)
                name: name of the transformer.
                keep_columns: keep original column names in the graph.
                **kwargs: kwargs to the parent constructor

            ..# noqa: I401
            ..# noqa: I402

            """
            self.name = name
            self.keep_columns = keep_columns
            # self.name = kwargs.pop("name", None)
            # logging.warning(
            #     "name is a deprecated kwarg. Please remove "
            #     "it from the kwargs and instead set it "
            #     "after instantiation."
            # )
            # self.keep_column = kwargs.pop("keep_columns", False)
            # logging.warning(
            #     "keep_columns is a deprecated kwarg. Please "
            #     "remove it from the kwargs and instead set "
            #     "it after instantiation."
            # )
            try:
                super(DFTransformer, self).__init__(*args, **kwargs)
            except TypeError as e:
                raise type(e)(
                    str(e) + ". Init for transformer: '{}' "
                    "called".format(transformer)
                )

            self.is_wrapped = True

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
                if getattr(self, "cache_manager", None) is not None:  # only
                    # used when part of the Foreshadow flow.
                    for column in X:
                        self.cache_manager["graph", column] = graph
                else:
                    logging.debug(
                        "cache_manager is not set for: " "{}".format(self)
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
                if getattr(self, "cache_manager", None) is not None:  # only
                    # used when part of the Foreshadow flow.
                    for column in X:
                        self.cache_manager["graph", column] = graph
                else:
                    logging.debug(
                        "cache_manager is not set for: " "{}".format(self)
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
                if getattr(self, "cache_manager", None) is not None:  # only
                    # used when part of the Foreshadow flow.
                    for column in X:
                        self.cache_manager["graph", column] = graph
                else:
                    logging.debug(
                        "cache_manager is not set for: " "{}".format(self)
                    )
            return out

        def __repr__(self):
            return "DF{}".format(self.__class__.__name__)

        @classmethod
        def _get_param_names(cls):
            """Shadow the parent __init__ method.

            Returns:
                _param_names for the parent class (and therefore the __init__).

            """
            return transformer._get_param_names()

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
