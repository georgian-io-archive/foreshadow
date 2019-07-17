"""Foreshadow extension of feature union for handling dataframes"""

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.externals.joblib import Parallel, delayed
from sklearn.pipeline import (
    FeatureUnion,
    _fit_one_transformer,
    _fit_transform_one,
    _transform_one,
)


class ParallelProcessor(FeatureUnion):
    """Class to support parallel operation on dataframes.

    This class functions similarly to a FeatureUnion except it divides a given
    pandas dataframe according to the transformer definition in the constructor
    and transforms the defined partial dataframes using the given transformers.
    It then concatenates the outputs together.

    Internally the ParallelProcessor uses MultiIndex-ing to identify the column
    of origin for transformer operations that result in multiple columns.

    The outer index or 'origin' index represents the column used to create a
    calculated column or represents the leftmost column of a series of columns
    used to create a calculated
    column.

    By default the output contains both Index's to support pipeline usage and
    tracking for the Preprocessor. This can be suppressed.

    Parameters:
        collapse_index (bool): Boolean defining whether multi-index should be
            flattened
        n_jobs (int): See base class
        transformer_weights (:obj:`numpy.ndarray`): See base class
        transformer_list (list): List of transformer objects in form
            [(name, [cols], TransformerObject), ...]

    """

    def __init__(
        self,
        transformer_list,
        n_jobs=1,
        transformer_weights=None,
        collapse_index=False,
    ):

        self.collapse_index = collapse_index
        self.default_transformer_list = None

        for item in transformer_list:
            self._set_names(item)

        super(ParallelProcessor, self).__init__(
            transformer_list, n_jobs, transformer_weights
        )

    def get_params(self, deep=True):
        """Return parameters of internal transformers.

        See :class:`sklearn.pipeline.FeatureUnion`

        Args:
            deep (bool): If True, will return the parameters for this estimator
                and contained subobjects that are estimators.

        Returns:
            dict: Parameter names mapped to their values.

        """
        self.default_transformer_list = [
            (a, b) for a, b, c in self.transformer_list
        ]
        return self._get_params("default_transformer_list", deep=deep)

    def set_params(self, **kwargs):
        """Set parameters of internal transformers.

        See :class:`sklearn.pipeline.FeatureUnion`

        Args:
            **kwargs: valid params of transformer to set

        Returns:
            self

        """
        self.default_transformer_list = [
            (a, b) for a, b, c in self.transformer_list
        ]
        return self._set_params("default_transformer_list", **kwargs)

    def _set_names(self, item):
        """Set internal names of transformers.

        Uses names defined in transformers list.

        Args:
            item: (name, transformer) tuple

        """
        # Sets name if name attribute exists
        if hasattr(item[1], "name"):
            item[1].name = item[0]
        # If steps attribute exists set names within all transformers
        if hasattr(item[1], "steps"):
            for step in item[1].steps:
                self._set_names(step)
        # If transformer_list exists set names within transformers_list
        if hasattr(item[1], "transformer_list"):
            for trans in item[1].transformer_list:
                self._set_names(trans)

    def _update_transformer_list(self, transformers):
        """Update local transformers list.

        Args:
            transformers: 1D iterable of transformers
        """
        transformers = iter(transformers)
        self.transformer_list[:] = [
            (name, None if old is None else next(transformers), cols)
            for name, old, cols in self.transformer_list
        ]

    def _validate_transformers(self):
        """Validate fit and transform methods exist and names are unique.

        Raises:
            TypeError: if fit, fit_transform, or transform are not implemented

        """
        names, transformers, cols = zip(*self.transformer_list)

        # validate names
        self._validate_names(names)

        # validate estimators
        for t in transformers:
            if t is None:
                continue
            if not (
                hasattr(t, "fit") or hasattr(t, "fit_transform")
            ) or not hasattr(t, "transform"):
                raise TypeError(
                    "All estimators should implement fit and "
                    "transform. '%s' (type %s) doesn't" % (t, type(t))
                )

    def _iter(self):
        """Iterate transformers list.

        Returns:
            list(list): tuple of (name, cols, transformer object, and the \
                transformer weights (non-applicable here))

        """
        get_weight = (self.transformer_weights or {}).get

        return (
            (name, trans, cols, get_weight(name))
            for name, trans, cols in self.transformer_list
            if trans is not None
        )

    def _get_other_cols(self, X):
        """Get all columns that are not defined in a transformer.

        Only include those that exist in the input dataframe.

        Args:
            X: input DataFrame

        Returns:
            Set of columns in DataFrame not defined in transformer

        """
        full = set(list(X))
        partial = set(
            list(
                _slice_cols(
                    X,
                    [c for _, _, cols, _ in self._iter() for c in cols],
                    drop_level=False,
                )
            )
        )

        return list(full - partial)

    def fit(self, X, y=None, **fit_params):
        """Fit data on the set of transformers.

        Args:
            X (:obj:`pandas.DataFrame`): Input X data
            y (:obj: 'pandas.DataFrame'): labels Y for data
            **fit_params (dict): Parameters to apply to transformers when
                fitting

        Returns:
            self

        """
        self.transformer_list = list(self.transformer_list)
        self._validate_transformers()

        # Create a parallel process of fitting transformers
        transformers = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_transformer)(
                trans,
                _slice_cols(X, cols),
                y,
                **{**fit_params, **_inject_df(trans, X)}
            )
            for name, trans, cols, weight in self._iter()
        )

        self._update_transformer_list(transformers)

        return self

    def transform(self, X):
        """Transform data using internal transformers.

        Args:
            X (:obj:`pandas.DataFrame`): Input X data

        Returns:
            :obj:`pandas.DataFrame`: All transformations concatenated

        """
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_pandas_transform_one)(
                trans, weight, _slice_cols(X, cols), cols
            )
            for name, trans, cols, weight in self._iter()
        )

        # Iterates columns not specific in transformers
        Xo = X[self._get_other_cols(X)]
        if len(list(Xo)) > 0:
            # Create multi-index with same name
            if type(list(Xo)[0]) != tuple:
                Xo.columns = [list(Xo), list(Xo)]

            Xs += (Xo,)

        if not Xs:
            # All transformers are None
            return X[[]]
        else:
            Xs = pd.concat(Xs, axis=1)

        # Reduces the multi-index to a single index if specified
        if self.collapse_index:
            Xs.columns = Xs.columns.droplevel()
        return Xs

    def fit_transform(self, X, y=None, **fit_params):
        """Perform both a fit and a transform.

        Args:
            X (:obj:`pandas.DataFrame`): Input X data
            y (:obj: 'pandas.DataFrame'): labels Y for data
            **fit_params (dict): Parameters to apply to transformers when
                fitting

        Returns:
            :obj:`pandas.DataFrame`: All transformations concatenated

        """
        self._validate_transformers()

        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_pandas_fit_transform_one)(
                trans,
                weight,
                _slice_cols(X, cols),
                y,
                cols,
                **{**fit_params, **_inject_df(trans, X)}
            )
            for name, trans, cols, weight in self._iter()
        )

        if not result:
            # All transformers are None
            return X[[]]

        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)

        Xo = X[self._get_other_cols(X)]

        # Iterates columns not being transformed
        if len(list(Xo)) > 0:
            # If a multi-index does not already exist create one with same
            # label
            if type(list(Xo)[0]) != tuple:
                Xo.columns = [list(Xo), list(Xo)]

            Xs += (Xo,)

        # Concatenate results
        Xs = pd.concat(Xs, axis=1)

        # Convert multi index to single index if specified
        if self.collapse_index:
            Xs.columns = Xs.columns.droplevel()
        return Xs


def _slice_cols(X, cols, drop_level=True):
    """Search for columns in dataframe using multi-index.

    Args:
        X (:obj:`pandas.DataFrame`): Input dataframe
        cols (list): List of cols to slice out of dataframe multi-index
        drop_level (bool): Whether to include the multi-index in the output

    Returns:
        :obj:`pd.DataFrame`: Data frame with sliced columns

    """
    # Get column list
    origin = list(X)

    # If no columns return the empty frame
    if len(origin) == 0:
        return X

    # If no columns are specified then drop all columns and return the empty
    # frame
    if len(cols) == 0:
        return X.drop(list(X), axis=1)

    # If a multi_index exists split it into origin (top) and new (bottom)
    if type(origin[0]) == tuple:
        origin, new = list(zip(*origin))
    # If a single index exists perform a simple slice and return
    else:
        return X[cols]

    # Utility function to perform the multi-index slice
    def get(c, level):
        ret = X.xs(c, axis=1, level=level, drop_level=False)
        if drop_level:
            ret.columns = ret.columns.droplevel()
        return ret

    # Iterate slice columns and use get to slice them out of the frame
    # Concatenate and return the result
    df = pd.concat(
        [
            get(c.replace("$", ""), "new") if c[0] == "$" else get(c, "origin")
            for c in cols
            if c in origin or c.replace("$", "") in new
        ],
        axis=1,
    )

    return df


def _inject_df(trans, df):
    """Insert temp parameters into fit_params dictionary.

    This is in case a transformer needs other columns for calculations or
    for hypothesis testing.

    Args:
        trans: transformer
        df: input dataframe

    Returns:
        params dict

    """
    return {
        "{}__full_df".format(k): df
        for k, v in trans.get_params().items()
        if isinstance(v, BaseEstimator)
    }


def _pandas_transform_one(transformer, weight, X, cols):
    """Transform dataframe using sklearn transformer then adds multi-index.

    Args:
        transformer: transformer
        weight: weighting for the one transformer
        X: input observations
        cols: columns for X

    Returns:
        output from _transform_one

    """
    colname = sorted(cols)[0]
    # Run original transform function
    res = _transform_one(transformer, weight, X)
    # Applies multi_index such that the id of the column set is the name of the
    # leftmost column in the list.
    res.columns = [[colname] * len(list(res)), list(res)]
    res.columns = res.columns.rename(["origin", "new"])
    return res


def _pandas_fit_transform_one(transformer, weight, X, y, cols, **fit_params):
    """Fit dataframe, executes transformation, then adds multi-index.

    Args:
        transformer: transformer to use
        weight: weight to use
        X: input observations
        y: input labels
        cols: column names as list
        **fit_params: params to transformer fit

    Returns:
        output from _fit_transform_one

    """
    colname = sorted(cols)[0]
    # Run original fit_transform function
    res, t = _fit_transform_one(transformer, weight, X, y, **fit_params)
    # Apply multi-index and name columns
    res.columns = [[colname] * len(list(res)), list(res)]
    res.columns = res.columns.rename(["origin", "new"])
    return res, t