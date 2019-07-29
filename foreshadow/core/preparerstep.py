"""General base classes used across Foreshadow."""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from foreshadow.core import logging
from foreshadow.transformers.core import ParallelProcessor
from foreshadow.transformers.core.notransform import NoTransform
from foreshadow.transformers.core.pipeline import DynamicPipeline


def _check_parallelizable_batch(column_mapping, group_number):
    """See if the group of cols 'group_number' is parallelizable.

    'group_number' in column_mapping is parallelizable if the cols across
    all steps are static. Then we can guarantee this Pipeline can run
    without needing any other step to finish first. 'group_number' is the
    outer key of the dict.

    Args:
        column_mapping: the column mapping from self.get_mapping()
        group_number: the group number
        PipelineClass: declare which Pipeline class to use. Default is a
            normal Pipeline but you can also use SingleInputPipeline.

    Returns:
        transformer_list if parallelizable, else None.

    """
    pipeline = column_mapping[group_number]
    if len(pipeline["inputs"]) == 1:
        inputs = pipeline["inputs"][0]
        steps = [(step.__class__.__name__, step) for step in pipeline["steps"]]
        # if we enter here, this step has the same columns across
        # all steps. This means that we can create one Pipeline for
        # this group of columns and let it run parallel to
        # everything else as its inputs are never dependent on the
        # result from any step in another pipeline.
        transformer_list = [
            "group: {}".format(group_number),
            DynamicPipeline(steps),
            inputs,
        ]
        # transformer_list = [name, pipeline of transformers, cols]
        # cols here is the same for each step, so we just pass it in
        # once as a single group.
        # TODO this is a very simple check, this could be optimized
        #  further
        return transformer_list
    else:
        # this group could not be separated out.
        return None


def _batch_parallelize(column_mapping, parallelized):
    """Batch parallelizes any groups in column_mapping if not parallelized.

    _check_parallelizable_batch will parallelize a group of columns across
    all steps of transformers if possible. The rest that are left have
    interdependencies and so the best we can do is to parallelize each step
    across all groups of columns. This helper performs that task and creates a
    Pipeline of steps that is parallelized across each group of cols at each
    step. This enabled format two of inputs, where columns can be shuffled
    around between steps.

    Args:
        column_mapping: the column_mapping from self.get_mapping()
        parallelized: mapping of group_number in column_mapping to True if
            already parallelized or False if not.

    Returns:
        list of steps for Pipeline, all_cols
        all_cols is the set of all cols that needs to be passed, as a list.

    Raises:
        ValueError: number inputs do not equal number of steps.

    """
    total_steps = len(column_mapping[0])
    steps = []  # each individual step, or dim1, will go in here.
    all_cols = set()
    for step_number in range(total_steps):
        groups = []
        for group_number in parallelized:
            if not parallelized[group_number]:  # we do not have a
                # transformer_list yet for this group.
                inputs = column_mapping["inputs"]
                steps = column_mapping["steps"]
                if len(inputs) != len(steps):
                    raise ValueError(
                        "number of inputs: {} does not equal "
                        "number of steps: {}".format(len(inputs), len(steps))
                    )
                list_of_steps = column_mapping[group_number]
                step_for_group = list_of_steps[step_number]
                transformer = step_for_group[0]
                cols = step_for_group[1]
                groups.append((group_number, transformer, cols))
                for col in cols:
                    all_cols.add(col)
        transformer_list = [
            [
                "group: {}, transformer: {}".format(group_number,
                                                    transformer.__name__),
                transformer,
                cols,
            ]
            for group_number, transformer, cols in groups
        ]  # this is one step parallelized across the columns (dim1
        # parallelized across dim2).
        steps.append(
            (
                "step: {}".format(step_number),
                ParallelProcessor(transformer_list, collapse_index=True),
            )
        )  # list of steps for final pipeline.
    return steps, list(all_cols)


class PreparerStep(BaseEstimator, TransformerMixin):
    """Base class for any pipeline step of DataPreparer.

    This class automatically wraps the defined pipeline to make it
    parallelizable across a DataFrame. To make this possible and still be
    customizable, subclasses must implement get_mapping, which tells this
    object how the columns would be grouped and what SmartTransformers will
    be used in what sequence. See get_mapping for instructions on how to
    implement it. This method automatically handles compiling steps into a
    usable format for ParallelProcessor and given mismatched columns,
    can handle that with the flag use_single_pipeline set to True.


    The transformer_list represents the mapping from columns to
    transformers, in the form of ['name', 'transformer', ['cols']],
    where the [cols] are the cols for transformer 'transformer. These
    transformers should be SmartTransformers for any subclass.

    The transformer_weights are multiplicative weights for features per
    transformer. Keys are transformer names, values the weights.

    """

    def __init__(
        self, column_sharer, *args, use_single_pipeline=False, **kwargs
    ):
        """Set the original pipeline steps internally.

        Takes a list of desired SmartTransformer steps and stores them as
        self._steps. Constructs self an sklearn pipeline object.

        Args:
            column_sharer: ColumnSharer instance to be shared across all steps.
            use_single_pipeline: Creates pipelines using SinglePipeline
                class instead of normal Pipelines.  .. #noqa: I102
            *args: args to Pipeline constructor.
            **kwargs: kwargs to PIpeline constructor.

        """
        self.column_sharer = column_sharer
        self._parallel_process = None
        self._use_single_pipeline = use_single_pipeline  # TODO right now,
        #  TODO this handles the case where a transformer outputs multiple
        #   columns and another transformer only accepts single inputs. We
        #   should make sure to define inputs and outputs for each
        #   transformer so that the Dynamic pipeline can match these
        #   requirements dynamically, at run time.
        super().__init__(*args, **kwargs)

    @staticmethod
    def separate_cols(transformers, X=None, cols=None):
        """Return a valid mapping where each col has a separate transformer.

        If X!=None, each individual col will create its own pipeline.
        Else, define the groups of cols that will have their pipeline by
        passing them into cols and leave X == None.

        Args:
            transformers: list of transformers of length equal to X.shape[1] or
                len(cols).
            X: input DataFrame. See description for when to pass.
            cols: DataFrame.columns, list of columns. See description for
                when to pass.

        Returns:
            A dict where each entry can be used to separately access an
            individual column from X.

        Raises:
            ValueError: input does not matched defined format.

        """
        if cols is None and X is not None:
            return {
                i: {"inputs": ([col],), "steps": transformers[i]}
                for i, col in enumerate(X)
            }
        elif X is None and cols is not None:
            return {
                i: {"inputs": (cols[i],), "steps": transformers[i]}
                for i, cols in enumerate(cols)
            }
        else:
            raise ValueError("Invalid input. Please read the docstring.")

    @classmethod
    def logging_name(cls):
        """Return the logging name for this transformer.

        Returns:
            See return.

        """
        return "DataPreparerStep: {} ".format(cls.__name__)

    def parallelize_mapping(self, column_mapping):
        """Create parallelized workflow for column_mapping.

        Each group of cols that is separated has the key: 'group_number' and a
        valid transformer_list for ParallelProcessor.
        The rest that are batch parallelized has key: 'grouped_pipeline' and
        a valid transformer_list for ParallelProcessor.

        Args:
            column_mapping: the column mapping returned from self.get_mapping()

        Returns:
            dict(), see above statement. Each value in this dict is a valid
            transformer_list for ParallelProcessor.

        """
        final_mapping = {}
        parallelized = {}  # we will first map all groups that have no
        # interdependencies with other groups. Then, we will do all the rest
        # of the groups after as they will be performed step-by-step
        # parallelized.
        for group_number in column_mapping:

            transformer_list = _check_parallelizable_batch(
                column_mapping, group_number
            )
            if transformer_list is None:  # could not be separated out
                parallelized[group_number] = False
            else:  # could be separated and parallelized
                final_mapping[group_number] = transformer_list
                parallelized[group_number] = True
        if len(final_mapping) < len(column_mapping) and False:  # then there
            # must be groups of columns that have interdependcies.
            # CURRENTLy DISABLED.
            steps, all_cols = _batch_parallelize(column_mapping, parallelized)
            final_mapping["grouped_pipeline"] = [
                "grouped_pipeline",
                Pipeline(steps),
                all_cols,
            ]

        return final_mapping

    def parallelize_smart_steps(self, X):
        """Make self.steps for internal pipeline methods parallelized.

        Takes self._steps passed at construction time and wraps each step
        with ParallelProcessor to parallelize it across a DataFrame's columns.
        Made possible sing get_transformer_list and get_transformer_weights
        which must be implemented by the subclass.

        get_transformer_list must return a mapping where each column shows
        up only once

        ['name', 'transformer', ['cols']] format where each column
            should be included exactly once.

        Args:
            X: DataFrame

        Returns:
            ParallelProcessor instance holding all the steps, parallelized
            as good as possible.

        """
        column_mapping = self.get_mapping(X)
        logging.debug(
            self.logging_name() + "column_mapping: {}".format(column_mapping)
        )
        logging.debug(self.logging_name() + "called ")
        parallelized_mapping = self.parallelize_mapping(column_mapping)
        group_transformer_list = [
            transformer_list
            for transformer_list in parallelized_mapping.values()
        ]
        if len(group_transformer_list) == 0:
            return NoTransform()
        return ParallelProcessor(group_transformer_list, collapse_index=True)

    def get_mapping(self, X):
        """Return dict of lists of tuples representing columns:transformers.

        The return can be viewed as a third order tensor, where:
        dim 1: the number of operations to be performed on a given set of
            columns. For instance, you could have this dimension = 3 where you
            would then view that a given column would have 3 Smart transformers
            associated with it.

        dim 2: the number of steps/operations. This can be viewed as
        groups of columns being passed to a single smart transformer. For
        instance, you may pass a single column each to its on smart
        transformer (say, to clean each column individually), or all columns
        to a single smart transformer (for instance, for dimensionality
        reduction).

        dim 3: The number of inputs to each SmartTransformer. Defines the
        width of the input space (the number of columns being passed).


        This data structure is constructed by using a nested dict structure, to
        more easily align with user configuration and serialization. The outer
        levels defines the number of groups of operations. Here, the key is not
        important but should be unique for each group. The second layer is a
        dict with two keys: 'inputs' and 'steps'. There are two accepted
        formats: inputs is a nested tuple of length 1, or nested tuple where
        the number of nested tuples is equal to the number of steps. In this
        case, each step is passed the inputs defined in each tuple. This
        latter case is not yet fully implemented.

        Of course, any SmartTranformer can be replaced with a concrete
        transformer, as a SmartTransformer is just a wrapper shadowing an
        underlying concrete transformer.

        Args:
            X: DataFrame

        Returns:
            third order list of lists, then None when finished.

        Raises:
            NotImplementedError: If child did not override and implement.

        .. #noqa: I202

        """
        raise NotImplementedError("Must implement this method.")

    def fit(self, X, *args, **kwargs):
        """Fit this step.

        calls underlying parallel process.

        Args:
            X: input DataFrame
            *args: args to _fit
            **kwargs: kwargs to _fit

        Returns:
            transformed data handled by Pipeline._fit

        """
        # TODO make fit remove a step if nothing is done, rather than a
        #  NoTransform Transformer.
        self.check_process(X)
        self._parallel_process.fit(X, *args, **kwargs)
        return self

    def check_process(self, X):
        """If fit was never called, makes sure to create the parallel process.

        Args:
            X: input DataFrame

        """
        if self._parallel_process is None:
            logging.debug(
                "DataPreparerStep: {} called check_process".format(
                    self.__class__.__name__)
            )
            self._parallel_process = self.parallelize_smart_steps(X)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit then transform this PreparerStep.

        calls underlying parallel process.

        Args:
            X: input DataFrame
            y: input labels
            **fit_params: kwarg params to fit

        Returns:
            Result from .transform()

        """
        self.check_process(X)
        return self._parallel_process.fit_transform(X, y=y, **fit_params)

    def transform(self, X, *args, **kwargs):
        """Transform X using this PreparerStep.

        calls underlying parallel process.

        Args:
            X: input DataFrame
            *args: args to .transform()
            **kwargs: kwargs to .transform()

        Returns:
            result from .transform()

        """
        return self._parallel_process.transform(X, *args, **kwargs)

    def inverse_transform(self, X, *args, **kwargs):
        """Inverse transform X using this PreparerStep.

        calls underlying parallel process.

        Args:
            X: input DataFrame.
            *args: args to .inverse_transform()
            **kwargs: kwargs to .inverse_transform()

        Returns:
            result from .inverse_transform()

        """
        self.check_process(X)
        return self._parallel_process.inverse_transform(X, *args, **kwargs)
