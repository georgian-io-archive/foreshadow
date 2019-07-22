"""Base classes for transformers."""

from abc import ABCMeta, abstractmethod

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.joblib import Parallel, delayed
from sklearn.pipeline import (
    FeatureUnion,
    Pipeline,
    _fit_one_transformer,
    _fit_transform_one,
    _transform_one,
)

from foreshadow.transformers.transformers import make_pandas_transformer
from foreshadow.utils import (
    check_df,
    get_transformer,
    is_transformer,
    is_wrapped,
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
            [(name, TransformerObject, [cols]), ...]

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


# TODO: Remove once _Empty is removed when DataCleaner is implemented
@make_pandas_transformer
class SmartTransformer(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    """Abstract transformer class for meta transformer selection decisions.

    This class contains the logic necessary to determine a single transformer
    or pipeline object that should act in its place.

    Once in a pipeline this class can be continuously re-fit in order to adapt
    to different data sets.

    Contains a function _get_tranformer that must be overridden by an
    implementing class that returns a scikit-learn transformer object to be
    used.

    Used and implements itself identically to a transformer.

    Attributes:
        override: A scikit-learn transformer that can be optionally provided
            to override internals logic. This takes top priority of all the
            setting.
        should_resolve: Whether or not the SmartTransformer will resolve
            the concrete transformer determination on each fit. This flag will
            set to `False` after the first fit. If force_reresolve is set, this
            will be ignored.
        force_reresolve: Forces re-resolve on each fit. If override is set it
            takes top priority.
        **kwargs: If overriding the transformer, these kwargs passed downstream
            to the overridden transformer

    """

    def __init__(
        self,
        y_var=False,
        override=None,
        should_resolve=True,
        force_reresolve=False,
        **kwargs
    ):
        self.kwargs = kwargs
        self.y_var = y_var
        self.transformer = None
        self.should_resolve = should_resolve
        self.force_reresolve = force_reresolve
        # Needs to be declared last as this overrides the resolve parameters
        self.override = override

    @property
    def transformer(self):
        """Get the selected transformer from the SmartTransformer.

        Returns:
            object: An instance of a concrete transformer.

        """
        return self._transformer

    @transformer.setter
    def transformer(self, value):
        """Validate transformer initialization.

        Args:
            value (object): The selected transformer that SmartTransformer
                should use.

        Raises:
            ValueError: If input is neither a valid foreshadow wrapped
                transformer, scikit-learn Pipeline, scikit-learn FeatureUnion,
                nor None.

        """
        # Check transformer type
        valid_tranformer = is_transformer(value) and is_wrapped(value)
        valid_pipeline = isinstance(value, Pipeline)
        valid_parallel = isinstance(value, FeatureUnion)
        set_none = value is None

        # Check the transformer inheritance status
        if (
            not valid_tranformer
            and not valid_pipeline
            and not valid_parallel
            and not set_none
        ):
            raise ValueError(
                "{} is neither a scikit-learn Pipeline, FeatureUnion, a "
                "wrapped foreshadow transformer, nor None.".format(value)
            )

        self._transformer = value

    @property
    def override(self):
        """Get the override parameter.

        Returns:
            str: The name of the override transformer class.

        """
        return self._override

    @override.setter
    def override(self, value):
        """Set the override parameter using a string class name.

        Args:
            value (str): The name of the desired transformer.

        """
        # self.name and self.keep_columns are injected as a result of pandas
        # wrapping. Try to resolve the transformer, otherwise error out.
        if value is not None:
            self.transformer = get_transformer(value)(
                name=self.name, keep_columns=self.keep_columns, **self.kwargs
            )
            self.should_resolve = False
            self.force_reresolve = False

        self._override = value

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Note: self.name and self.keep_columns are provided by the wrapping
            method

        Args:
            deep (bool): If True, will return the parameters for this estimator
                and contained sub-objects that are estimators.

        Returns:
            Parameter names mapped to their values.

        """
        return {
            "y_var": self.y_var,
            "override": self.override,
            "name": self.name,
            "keep_columns": self.keep_columns,
            **(
                self.transformer.get_params(deep=deep)
                if self.transformer is not None and deep
                else {}
            ),
        }

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with :meth:`get_params()`.

        Args:
            **params (dict): any valid parameter of this estimator

        """
        self.name = params.pop("name", self.name)
        self.keep_columns = params.pop("keep_columns", self.keep_columns)
        self.y_var = params.pop("y_var", self.y_var)

        # Calls to override auto set the transformer instance
        self.override = params.pop("override", self.override)

        if self.transformer is not None:
            valid_params = {
                k.partition("__")[2]: v
                for k, v in params.items()
                if k.split("__")[0] == "transformer"
            }
            self.transformer.set_params(**valid_params)

    @abstractmethod
    def pick_transformer(self, X, y=None, **fit_params):
        """Pick the correct transformer object for implementations.

        Args:
            X (:obj:`pandas.DataFrame`): Input X data
            y (:obj: 'pandas.DataFrame'): labels Y for data
            **fit_params (dict): Parameters to apply to transformers when
                fitting

        """
        pass  # pragma: no cover

    def resolve(self, X, y=None, **fit_params):
        """Verify transformers have the necessary methods and attributes.

        Args:
            X: input observations
            y: input labels
            **fit_params: params to fit

        """
        # If override is passed in or set, all types of resolves are turned
        # off.
        # Otherwise, force_reresolve will always resolve on each fit.

        # If force_reresolve is set, always re-resolve
        if self.force_reresolve:
            self.should_resolve = True

        # Only resolve if transformer is not set or re-resolve is requested.
        if self.should_resolve:
            self.transformer = self.pick_transformer(X, y, **fit_params)
            self.transformer.name = self.name
            self.transformer.keep_columns = self.keep_columns

        # reset should_resolve
        self.should_resolve = False

    def transform(self, X):
        """See base class.

        Args:
            X: transform

        Returns:
            transformed X using selected best transformer.

        """
        X = check_df(X)
        self.resolve(X)
        return self.transformer.transform(X)

    def fit(self, X, y=None, **fit_params):
        """See base class.

        Args:
            X: see base class
            y: see base class
            **fit_params: see base class

        Returns:
            see base class

        """
        X = check_df(X)
        y = check_df(y, ignore_none=True)
        self.resolve(X, y, **fit_params)
        self.transformer.full_df = fit_params.pop("full_df", None)

        return self.transformer.fit(X, y, **fit_params)

    def inverse_transform(self, X):
        """Invert transform if possible.

        Args:
            X: transformed input observations using selected best transformer

        Returns:
            original input observations

        """
        X = check_df(X)
        self.resolve(X)
        return self.transformer.inverse_transform(X)


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
