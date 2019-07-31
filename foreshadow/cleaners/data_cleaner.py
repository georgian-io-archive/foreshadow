"""Cleaner module for cleaning data as step in Foreshadow workflow."""
from collections import namedtuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from foreshadow.core import logging
from foreshadow.core.preparerstep import PreparerStep
from foreshadow.exceptions import InvalidDataFrame
from foreshadow.metrics.internals import avg_col_regex, regex_rows
from foreshadow.transformers.core import SmartTransformer
from foreshadow.transformers.internals.notransform import NoTransform
from foreshadow.utils.testing import dynamic_import
from foreshadow.utils.validation import check_df


CleanerReturn = namedtuple("CleanerReturn", ["row", "match_lens"])


class DataCleaner(PreparerStep):
    """Determine and perform best data cleaning step."""

    def __init__(self, **kwargs):
        """Define the single step for DataCleaner, using SmartCleaner.

        Args:
            **kwargs: kwargs to PreparerStep constructor.

        """
        super().__init__(use_single_pipeline=True, **kwargs)

    def get_mapping(self, X):
        """Return the mapping of transformations for the DataCleaner step.

        Args:
            X: input DataFrame.

        Returns:
            Mapping in accordance with super.

        """
        return self.separate_cols(
            transformers=[
                [
                    SmartFlatten(column_sharer=self.column_sharer),
                    SmartCleaner(column_sharer=self.column_sharer),
                ]
                for c in X
            ],
            cols=X.columns,
        )

    def __repr__(self):
        """Return string representation of this object with parent params.

        Returns:
            See above.

        """
        r = super().__repr__()
        preparer_params = self._preparer_params()
        preparer_params = {p: getattr(self, p, None) for p in preparer_params}
        preparer_print = ", ".join(
            ["{}={}".format(k, v) for k, v in preparer_params.items()]
        )
        return r[:-1] + preparer_print + ")"


class SmartCleaner(SmartTransformer):
    """Intelligently decide which cleaning function should be applied."""

    def __init__(
        self,  # manually adding as otherwise get_params won't see it.
        check_wrapped=False,
        **kwargs
    ):
        self.single_input = True  # all transformers under this only accept
        # 1 column. This is how DynamicPipeline knows this.
        # get_params then set_params, it may be in kwargs already
        super().__init__(check_wrapped=check_wrapped, **kwargs)

    def pick_transformer(self, X, y=None, **fit_params):
        """Get best transformer for a given column.

        Args:
            X: input DataFrame
            y: input labels
            **fit_params: fit_params

        Returns:
            Best data cleaning transformer.

        """
        from foreshadow.cleaners.internals import __all__ as cleaners

        cleaners = [
            (dynamic_import(cleaner, "foreshadow.cleaners.internals"), cleaner)
            for cleaner in cleaners
            if cleaner.lower().find("cleaner") != -1
        ]
        best_score = 0
        best_cleaner = None
        for cleaner, name in cleaners:
            cleaner = cleaner(column_sharer=self.column_sharer)
            score = cleaner.metric_score(X)
            if score > best_score:
                best_score = score
                best_cleaner = cleaner
        if best_cleaner is None:
            return NoTransform(column_sharer=self.column_sharer)
        return best_cleaner

    def resolve(self, X, *args, **kwargs):
        """Resolve the underlying concrete transformer.

        Sets self.column_sharer with the domain tag.

        Args:
            X: input DataFrame
            *args: args to super
            **kwargs: kwargs to super

        Returns:
            Return from super.

        """
        ret = super().resolve(X, *args, **kwargs)
        if self.column_sharer is not None:
            self.column_sharer[
                "domain", X.columns[0]
            ] = self.transformer.__class__.__name__
        else:
            logging.debug("column_sharer was None")
        return ret


class SmartFlatten(SmartTransformer):
    """Smartly determine how to flatten an input DataFrame."""

    def __init__(self, check_wrapped=True, **kwargs):
        super().__init__(check_wrapped=check_wrapped, **kwargs)

    def pick_transformer(self, X, y=None, **fit_params):
        """Get best transformer for a given column.

        Args:
            X: input DataFrame
            y: input labels
            **fit_params: fit_params

        Returns:
            Best data flattening transformer

        """
        from foreshadow.cleaners.internals import __all__ as flatteners

        flatteners = [
            (
                dynamic_import(flattener, "foreshadow.cleaners.internals"),
                flattener,
            )
            for flattener in flatteners
            if flattener.lower().find("flatten") != -1
        ]

        best_score = 0
        best_flattener = None
        for flattener, name in flatteners:
            flattener = flattener(column_sharer=self.column_sharer)
            score = flattener.metric_score(X)
            if score > best_score:
                best_score = score
                best_flattener = flattener
        if best_flattener is None:
            return NoTransform(column_sharer=self.column_sharer)
        return best_flattener

    def resolve(self, X, *args, **kwargs):
        """Resolve the underlying concrete transformer.

        Sets self.column_sharer with the domain tag.

        Args:
            X: input DataFrame
            *args: args to super
            **kwargs: kwargs to super

        Returns:
            Return from super.

        """
        ret = super().resolve(X, *args, **kwargs)
        if self.column_sharer is not None:
            self.column_sharer[
                "domain", X.columns[0]
            ] = self.transformer.__class__.__name__
        else:
            logging.debug("column_sharer was None")
        return ret


class BaseCleaner(BaseEstimator, TransformerMixin):
    """Base class for any Cleaner Transformer."""

    def __init__(
        self,
        transformations,
        output_columns=None,
        confidence_computation=None,
        default=lambda x: x,
        column_sharer=None,
    ):
        """Construct any cleaner/flattener.

        Args:
            transformations: a callable that takes a string and returns a
                tuple with the length of the transformed characters and then
                transformed string.
            output_columns: If none, any lists returned by the transformations
                are assumed to be separate columns in the new DataFrame.
                Otherwise, pass the names for each desired output
                column to be used.
            confidence_computation: The dict of {metric: weight} for the
                subclass's metric computation. This implies an OVR model.
            default: Function that returns the default value for a row if
                the transformation failed. Accepts the row as input.
            column_sharer: An instance of 'foreshadow.core.ColumnSharer' to
                be used to share information between PreparerSteps.

        Raises:
            ValueError: If not a list, int, or None specifying expected
                output columns.

        """
        if not isinstance(output_columns, (int, list, type(None))):
            raise ValueError("output columns not a valid type")

        self.default = default
        self.output_columns = output_columns
        self.transformations = transformations
        self.confidence_computation = {regex_rows: 0.8, avg_col_regex: 0.2}
        self.column_sharer = column_sharer
        if confidence_computation is not None:
            self.confidence_computation = confidence_computation

    def metric_score(self, X):
        """Compute the score for this cleaner using confidence_computation.

        confidence_computation is passed through init for each subclass.
        The confidence determines which cleaner/flattener is picked in an
        OVR fashion.

        Args:
            X: input DataFrame.

        Returns:
            float: confidence value.

        """
        return sum(
            [
                metric_fn(X, cleaner=self.transform_row) * weight
                for metric_fn, weight in self.confidence_computation.items()
            ]
        )

    def transform_row(self, row_of_feature, return_tuple=True):
        """Perform clean operations on text, that is a row of feature.

        Uses self.transformations determined at init time by the child class
        and performs the transformations sequentially.

        Args:
            row_of_feature: one row of one column
            return_tuple: return named_tuple object instead of just the row.
                This will often be set to False when passing this method to an
                external function (non source code) that will expect the
                output to only be the transformed row, such as DataFrame.apply.

        Returns:
            NamedTuple object with:
            .text
            the text in row_of_feature transformed by transformations. If
            not possible, it will be None.
            .match_lens
            the number of characters from original text at each step that
            was transformed.

        """
        matched_lengths = []  # this does not play nice with creating new
        # columns
        for transform in self.transformations:
            row = row_of_feature
            row, match_len = transform(row)
            if match_len == 0:
                matched_lengths.append(0)
                row = self.default(row_of_feature)
                break
            matched_lengths.append(match_len)
        if return_tuple:
            return CleanerReturn(row, matched_lengths)
        else:
            return row

    def fit(self, X, y=None):
        """Empty fit.

        Args:
            X: input observations
            y: input labels

        Returns:
            self

        """
        return self

    def transform(self, X, y=None):
        """Clean string columns.

        Here, we assume that any list output means that these are desired
        to be new columns in our dataset. Contractually, this could change
        to be that a boolean flag is passed to indicate when this is
        desired, as of right now, there should be no need to return a list
        for any case other than this case of desiring new column.

        The same is assumed for dicts, where the key is the new column name,
        the value is the value for that row in that column. NaNs
        are automatically put into the columns that don't exist for given rows.

        Args:
            X (:obj:`pandas.Series`): X data
            y: input labels

        Returns:
            :obj:`pandas.DataFrame`: Transformed data

        Raises:
            InvalidDataFrame: If unexpected output returned that was not
                handled correctly. This happens if the output specified by the
                child does not match what is actually returned. The child
                should ensure it's implementation is consistent.

        """
        X = check_df(X, single_column=True)
        # Problem:
        # I can use .apply to perform all these transformations and that
        # works beautifully, except when I want to define a funtion that
        # will use the pandas.series.str.split operation. In which case,
        # the .apply fails and I don't know why.

        # I need each function to accept the row as an argument so that we
        # can inspect how much of the text was matched (for determining if
        # it should be used). however, doing this means I need to iterate
        # over each row for a given column on my own, which requires me to
        # leave

        out = X[X.columns[0]].apply(self.transform_row, return_tuple=False)
        # access single column as series and apply the list of
        # transformations to each row in the series.
        if any(
            [
                isinstance(out.iloc[i], (list, tuple))
                for i in range(out.shape[0])
            ]
        ):  # out are lists == new columns
            if not all(
                [
                    len(out.iloc[0]) == len(out.iloc[i])
                    for i in range(len(out.iloc[0]))
                ]
            ):
                raise InvalidDataFrame(
                    "length of lists: {}, returned not of same value.".format(
                        [out.iloc[i] for i in range(len(out[0]))]
                    )
                )
            columns = self.output_columns
            if columns is None:
                columns = [
                    X.columns[0] + str(c) for c in range(len(out.iloc[0]))
                ]
                # by default, pandas would have given a unique integer to
                # each column, instead, we keep the previous column name and
                # add that integer.
            X = pd.DataFrame([*out.values], columns=columns)
        elif any(
            [isinstance(out.iloc[i], (dict)) for i in range(out.shape[0])]
        ):  # out are dicts ==  named new columns
            all_keys = dict()
            for row in out:
                all_keys.update({key: True for key in row})  # get all columns
            columns = list(all_keys.keys())
            out = pd.DataFrame([*out.values], columns=columns)
            out.columns = [X.columns[0] + "_" + c for c in columns]
            X = out
            # by default, this will create a DataFrame where if a row
            # contains the value, it will be added, if not NaN is added.
        else:  # no lists, still 1 column output
            X[X.columns[0]] = out
        return X
