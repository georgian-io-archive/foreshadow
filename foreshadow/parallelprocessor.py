"""Foreshadow extension of feature union for handling dataframes."""

import pandas as pd
from sklearn.externals.joblib import Parallel, delayed
from sklearn.pipeline import (
    FeatureUnion,
    _fit_one_transformer,
    _fit_transform_one,
    _transform_one,
)

from foreshadow.base import BaseEstimator
from foreshadow.logging import logging
from foreshadow.utils.common import ConfigureCacheManagerMixin

from .serializers import PipelineSerializerMixin, _make_serializable


class ParallelProcessor(
    FeatureUnion, PipelineSerializerMixin, ConfigureCacheManagerMixin
):
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
        self.processing_step_name = self.__class__.__name__

        for item in transformer_list:
            self._set_names(item)

        super(ParallelProcessor, self).__init__(
            transformer_list, n_jobs, transformer_weights
        )

    def configure_step_name(self, name):
        """Configure the processing step name of this parallel processor.

        Args:
            name: step name

        """
        self.processing_step_name = name

    def dict_serialize(self, deep=False):
        """Serialize the selected params of parallel_process.

        Args:
            deep (bool): see super

        Returns:
            dict: parallel_process serialized in customized form.

        """
        params = self.get_params(deep=False)
        params["transformer_list"] = self._convert_transformer_list(
            params["transformer_list"]
        )

        return _make_serializable(params, serialize_args=self.serialize_params)

    def configure_cache_manager(self, cache_manager):
        """Configure cache_manager in each dynamic pipeline of the transformer_list.

        Args:
            cache_manager: a cache_manager instance

        """
        for transformer_triple in self.transformer_list:
            dynamic_pipeline = transformer_triple[1]
            for step in dynamic_pipeline.steps:
                step[1].cache_manager = cache_manager

    @staticmethod
    def _convert_transformer_list(transformer_list):
        """Convert the transformer list into a desired form.

        Initially the transformer list has a form of
        [("group_num", dynamic_pipeline, ["col1", "col2", ...]), ...].

        We convert it into a form of
        [{"col": {"pipeline": dynamic_pipeline}}, ...] for a single column
        or
        [{"group: GROUP_NAME": {"pipeline": dynamic_pipeline,
        "columns":["col1", "col2", ...]}}]

        Args:
            transformer_list: the transformer list in the parallel_processor

        Returns:
            list: converted transformer list.

        """
        result = []
        for transformer_triple in transformer_list:
            converted = {}
            group_name, dynamic_pipeline, column_group = transformer_triple
            if len(column_group) > 1:
                converted[group_name] = {
                    "processing_pipeline": dynamic_pipeline,
                    "columns": column_group,
                }
            else:
                converted[",".join(column_group)] = {
                    "processing_pipeline": dynamic_pipeline
                }
            result.append(converted)
        return result

    @classmethod
    def reconstruct_parallel_process(cls, data):
        """Reconstruct the parallel process in a preparestep.

        Args:
            data: serialized block of a parallel process

        Returns:
            a reconstructed parallel processor

        """
        if (
            "transformation_by_column_group" not in data
            or data["transformation_by_column_group"] is None
        ):
            return None

        n_jobs = data["n_jobs"]
        transformer_weights = data["transformer_weights"]
        collapse_index = data["collapse_index"]

        transformer_list = []
        for i, transformation in enumerate(
            data["transformation_by_column_group"]
        ):
            transformer_list.append(
                (
                    cls.__extract_elements_for_parallel_process(
                        i, transformation
                    )
                )
            )

        return ParallelProcessor(
            transformer_list, n_jobs, transformer_weights, collapse_index
        )

    @classmethod
    def __extract_elements_for_parallel_process(cls, i, transformation):
        item = list(transformation.values())[0]
        dynamic_pipeline = item["processing_pipeline"]
        if "columns" in item:
            column_groups = item["columns"]
            group_name = list(transformation.keys())[0]
        else:
            column_groups = list(transformation.keys())[0].split(",")
            group_name = "group: {}".format(str(i))
        return group_name, dynamic_pipeline, column_groups

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
        # TODO this method may need to add multiprocess cache_manager
        #  updates if we decide to use it somewhere in the code. Currently
        #  it is only used in a unit test.
        self.transformer_list = list(self.transformer_list)
        self._validate_transformers()

        # Create a parallel process of fitting transformers
        transformers = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_transformer)(
                trans, _slice_cols(X, cols), y, **fit_params
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
                trans, weight, _slice_cols(X, cols), cols, self.collapse_index
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
            # if self.collapse_index:
            #     Xs = pd.concat([Xs[i].get_level_values('new') for i in
            #                     range(len(Xs))], axis=1)
            # else:
            Xs = pd.concat(Xs, axis=1)
        # Reduces the multi-index to a single index if specified
        if self.collapse_index:
            try:
                Xs.columns = Xs.columns.droplevel()
                Xs.index.name = None
                Xs.columns.name = None
            except AttributeError:  # TODO figure out why is this needed
                pass
        return Xs

    def _get_original_cache_manager(self):
        _, transformers, _ = zip(*self.transformer_list)
        for transformer in transformers:
            # in case the transformer does not have steps
            if hasattr(transformer, "steps"):
                for step in transformer.steps:
                    # steps like Imputer don't have cache_manager
                    if hasattr(step[1], "cache_manager"):
                        return step[1].cache_manager
            elif hasattr(transformer, "cache_manager"):
                return transformer.cache_manager
        return None

    @staticmethod
    def _update_original_cache_manager(cache_manager, transformers):
        for transformer in transformers:
            if hasattr(transformer, "steps"):
                modified_cs = transformer.steps[0][1].cache_manager
            elif hasattr(transformer, "cache_manager"):
                modified_cs = transformer.cache_manager
            if modified_cs:
                ParallelProcessor._update_original_cache_manager_with_another(
                    cache_manager, modified_cs
                )

    @staticmethod
    def _update_transformers_with_updated_cache_manager(
        transformers, cache_manager
    ):
        for transformer in transformers:
            if hasattr(transformer, "steps"):
                for step in transformer.steps:
                    step[1].cache_manager = cache_manager
            elif hasattr(transformer, "cache_manager"):
                transformer.cache_manager = cache_manager

    @staticmethod
    def _update_original_cache_manager_with_another(
        cache_manager, modified_cs
    ):
        """Update the cache_manager with another cache_manager in place.

        Only values that are not None are assigned back to the cache_manager.

        Args:
            cache_manager: the original cache_manager.
            modified_cs: a modified cache_manager by a parallel_process.

        """
        for combined_key in modified_cs:
            if modified_cs[combined_key] is not None:
                cache_manager[combined_key] = modified_cs[combined_key]

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

        cache_manager = self._get_original_cache_manager()
        # TODO not all preparesteps need to update the cache_manager. This
        #  is something we may be able to improve by specifying the
        #  update_cache_manager params through the fit_params (we need to
        #  pop it).
        update_cache_manager = (self.n_jobs > 1 or self.n_jobs == -1) and (
            cache_manager is not None
        )

        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_pandas_fit_transform_one)(
                trans,
                weight,
                _slice_cols(X, cols),
                y,
                cols,
                self.collapse_index,
                self.processing_step_name,
                **fit_params,
            )
            for name, trans, cols, weight in self._iter()
        )
        if not result:
            # All transformers are None
            return X[[]]

        Xs, transformers = zip(*result)
        if update_cache_manager:
            self._update_original_cache_manager(cache_manager, transformers)
            self._update_transformers_with_updated_cache_manager(
                transformers, cache_manager
            )
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
        result = pd.concat(Xs, axis=1)

        # Convert multi index to single index if specified
        if self.collapse_index:
            try:
                result.columns = result.columns.droplevel()
                result.index.name = None
                result.columns.name = None
            except AttributeError:  # TODO figure out why this is needed
                pass
        return result

    def inverse_transform(self, X, **inverse_params):
        """Perform both a fit and a transform.

        Inverse transform should not update the cache_manager as it is
        only used in prediction in case the target column has been
        transformed during the training process. Once the model is trained,
        the cache_manager should not be touched during prediction process.

        Args:
            X (:obj:`pandas.DataFrame`): Input X data
            **inverse_params (dict): Parameters to apply to transformers when
                inverse transforming

        Returns:
            :obj:`pandas.DataFrame`: All transformations concatenated

        """
        self._validate_transformers()

        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_pandas_inverse_transform_one)(
                trans,
                weight,
                _slice_cols(X, cols),
                cols,
                self.collapse_index,
                **inverse_params,
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
            try:
                Xs.columns = Xs.columns.droplevel()
                Xs.index.name = None
                Xs.columns.name = None
            except AttributeError:  # TODO figure out why this is needed
                pass
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


def _pandas_transform_one(transformer, weight, X, cols, collapse_index):
    """Transform dataframe using sklearn transformer then adds multi-index.

    Args:
        transformer: transformer
        weight: weighting for the one transformer
        X: input observations
        cols: columns for X
        collapse_index: collapse multi-index to single-index

    Returns:
        output from _transform_one

    """
    colname = sorted(cols)[0]
    # Run original transform function
    res = _transform_one(transformer, weight, X)
    # Applies multi_index such that the id of the column set is the name of the
    # leftmost column in the list.
    if not collapse_index:
        res.columns = [[colname] * len(list(res)), list(res)]
        res.columns = res.columns.rename(["origin", "new"])
    return res


def _pandas_fit_transform_one(
    transformer, weight, X, y, cols, collapse_index, step_name="", **fit_params
):
    """Fit dataframe, executes transformation, then adds multi-index.

    Args:
        transformer: transformer to use
        weight: weight to use
        X: input observations
        y: input labels
        cols: column names as list
        collapse_index: collapse multi-index to single-index
        step_name: name of the processing step owning the parallel processor
        **fit_params: params to transformer fit

    Returns:
        output from _fit_transform_one

    """
    logging_template = "{} processing an individual column [{}]"
    if len(cols) > 1:
        logging_template = "{} processing a group of columns [{}]"
    logging.info(logging_template.format(step_name, ",".join(map(str, cols))))
    colname = sorted(cols)[0]
    # Run original fit_transform function
    res, t = _fit_transform_one(transformer, weight, X, y, **fit_params)
    # Apply multi-index and name columns
    if not collapse_index:
        res.columns = [[colname] * len(list(res)), list(res)]
        res.columns = res.columns.rename(["origin", "new"])
    return res, t


def _inverse_transform_one(transformer, weight, X, **inverse_params):
    res = transformer.inverse_transform(X, **inverse_params)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res, transformer
    return res * weight, transformer


def _pandas_inverse_transform_one(
    transformer, weight, X, cols, collapse_index, **inverse_params
):
    """Inverse_transform DF then adds multi-index.

    Args:
        transformer: transformer to use
        weight: weight to use
        X: input observations
        cols: column names as list
        collapse_index: collapse multi-index to single-index
        **inverse_params: params to transformer inverse_transform

    Returns:
        output from _inverse_transform_one

    """
    colname = sorted(cols)[0]
    # Run original fit_transform function
    res, t = _inverse_transform_one(transformer, weight, X, **inverse_params)
    # Apply multi-index and name columns
    if not collapse_index:
        res.columns = [[colname] * len(list(res)), list(res)]
        res.columns = res.columns.rename(["origin", "new"])
    return res, t
