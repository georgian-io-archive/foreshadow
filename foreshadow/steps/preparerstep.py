"""General base classes used across Foreshadow."""
from collections import MutableMapping, defaultdict, namedtuple

from foreshadow.base import BaseEstimator, TransformerMixin
from foreshadow.concrete.internals.notransform import NoTransform
from foreshadow.config import config
from foreshadow.logging import logging
from foreshadow.parallelprocessor import ParallelProcessor
from foreshadow.serializers import _make_deserializable
from foreshadow.utils.common import ConfigureColumnSharerMixin

from ..columnsharer import ColumnSharer
from ..pipeline import DynamicPipeline
from ..serializers import ConcreteSerializerMixin


GroupProcess = namedtuple(
    "GroupProcess", ["single_input", "step_inputs", "steps"]
)


class PreparerMapping(MutableMapping):
    """Mapping to be returned by any subclass of PreparerStep.

    This mapping is a dict of namedtuples used internally by
    PreparerStep and should be created by using the .add() method.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.store = defaultdict(lambda: defaultdict(lambda: None))

    def __getitem__(self, group_number):
        """Get item by internal group number, the parallel process number.

        Args:
            group_number: auto-incrementing integer field set internally.

        Returns:
            GroupProcess for group_number.

        """
        return self.store[group_number]  # TODO make this as well accept by
        # TODO list of columns representing a grouping, though this will have
        #  decreased efficiency.

    def __setitem__(self, group_number, group_process):
        """Set new GroupProcess to key group_number.

        Args:
            group_number: the group_number for this process. Should be
                unique. Suggested to be auto-incrementing.
            group_process: the GroupProcess namedtuple for this group_number.

        """
        self.store[group_number] = group_process

    def __delitem__(self, group_number):
        """Enable deletion by column or by key.

        Args:
            group_number: the internal 'group_number' to delete.

        """
        del self.store[group_number]

    def __iter__(self):
        """Iteratore over group_processes.

        Returns:
            Iterator over internal dict.

        """
        # handle nested
        return iter(self.store)

    def __len__(self):
        """Return number of processes.

        Returns:
            Number of processes.

        """
        return len(self.store)

    def add(self, inputs, transformers, group_name):
        """Add another group_process defined by inputs and transformers.

        Main API method to be used by subclasses of PreparerStep.

        Args:
            inputs: the input columns. May be a list of columns, or a list
                of lists representing the groups of columns for each
                transformer.
            transformers: the transformers.
            group_name: the name of the group

        Raises:
            ValueError: if invalid input format.

        """
        if isinstance(inputs, (str, int)):
            logging.warning(
                "Input column converted to proper list format. "
                "This automatic inspection may not have the "
                "desired so effect, so please follow "
                "PreparerMapping.add() input."
            )
            inputs = [inputs]
        if not isinstance(inputs[0], (list, tuple)):
            # one set of inputs at the beginning.
            self[group_name] = GroupProcess(inputs, None, transformers)
            # leverage setitem of self.
        elif len(inputs) == len(transformers):  # defined inputs for each
            # transformer
            self[group_name] = GroupProcess(None, inputs, transformers)
            # leverage setitem of self.
        else:
            raise ValueError(
                "inputs: {} do no match valid options for "
                "transformers.".format(inputs)
            )


def _check_parallelizable_batch(group_process, group_name):
    """See if the group of cols 'group_number' is parallelizable.

    'group_number' in column_mapping is parallelizable if the cols across
    all steps are static. Then we can guarantee this Pipeline can run
    without needing any other step to finish first. 'group_number' is the
    outer key of the dict.

    Args:
        group_process: an item from self.get_mapping(), a GroupProcess
            namedtupled
        group_name: the group name

    Returns:
        transformer_list if parallelizable, else None.

    """
    if group_process.single_input is not None:
        inputs = group_process.single_input
        if group_process.steps is not None:
            steps = [
                (step.__class__.__name__, step) for step in group_process.steps
            ]
        else:
            steps = None
        # if we enter here, this step has the same columns across
        # all steps. This means that we can create one Pipeline for
        # this group of columns and let it run parallel to
        # everything else as its inputs are never dependent on the
        # result from any step in another pipeline.
        transformer_list = [
            "group: {}".format(group_name),
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


def _batch_parallelize(column_mapping):
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
        for group_name, group_process in column_mapping:
            if group_process.step_inputs is not None:  # we do not have a
                # transformer_list yet for this group.
                inputs = column_mapping["inputs"]
                steps = column_mapping["steps"]
                if len(inputs) != len(steps):
                    raise ValueError(
                        "number of inputs: {} does not equal "
                        "number of steps: {}".format(len(inputs), len(steps))
                    )
                list_of_steps = column_mapping[group_name]
                step_for_group = list_of_steps[step_number]
                transformer = step_for_group[0]
                cols = step_for_group[1]
                groups.append((group_name, transformer, cols))
                for col in cols:
                    all_cols.add(col)
        transformer_list = [
            [
                "group: {}, transformer: {}".format(
                    group_name, transformer.__name__
                ),
                transformer,
                cols,
            ]
            for group_name, transformer, cols in groups
        ]  # this is one step parallelized across the columns (dim1
        # parallelized across dim2).
        steps.append(
            (
                "step: {}".format(step_number),
                ParallelProcessor(
                    transformer_list,
                    n_jobs=config.get_n_jobs_config(),
                    collapse_index=True,
                ),
            )
        )  # list of steps for final pipeline.
    return steps, list(all_cols)


class PreparerStep(
    BaseEstimator,
    TransformerMixin,
    ConcreteSerializerMixin,
    ConfigureColumnSharerMixin,
):
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

    def __init__(self, column_sharer=None, **kwargs):
        """Set the original pipeline steps internally.

        Takes a list of desired SmartTransformer steps and stores them as
        self._steps. Constructs self an sklearn pipeline object.

        Args:
            column_sharer: ColumnSharer instance to be shared across all steps.
            use_single_pipeline: Creates pipelines using SingleInputPipeline
                class instead of normal Pipelines.  .. #noqa: I102
            **kwargs: kwargs to PIpeline constructor.

        """
        self._parallel_process = None
        if "_parallel_process" in kwargs:  # clone will try to init using
            # the params from get_params, meaning this will be passed
            # through even though its not a part of the init.
            self._parallel_process = kwargs.pop("_parallel_process")
        self.column_sharer = column_sharer
        if self.column_sharer is None:
            self.column_sharer = ColumnSharer()
        super().__init__(**kwargs)

    def configure_column_sharer(self, column_sharer):
        """Recursively configure column sharer attribute.

        Args:
            column_sharer:  a column sharer instance.

        """
        super().configure_column_sharer(column_sharer)
        self._parallel_process.configure_column_sharer(column_sharer)

    def dict_serialize(self, deep=False):
        """Serialize the preparestep.

        It renames transformer_list to transformation_by_column_group.

        Args:
            deep: see super

        Returns:
            a serialized preparestep.

        """
        serialized = super().dict_serialize(deep=deep)["_parallel_process"]
        transformer_list = serialized["transformer_list"]
        serialized.pop("transformer_list")
        serialized["transformation_by_column_group"] = transformer_list
        return serialized

    @classmethod
    def dict_deserialize(cls, data):
        """Deserialize the transformer by reconstructing the parallel processor.

        Args:
            data: serialized preparestep

        Returns:
            a reconstructed preparestep

        """
        params = _make_deserializable(data)
        parallel_processor = ParallelProcessor.reconstruct_parallel_process(
            params
        )
        reconstructed_params = {"_parallel_process": parallel_processor}
        ret_tf = cls()
        ret_tf.set_params(**reconstructed_params)
        return ret_tf

    @staticmethod
    def separate_cols(transformers, cols, criterion=None):
        """Return a valid mapping where each col has a separate transformer.

        Define the groups of cols that will have their pipeline by
        passing them into cols. If simply X.columns, each individual column
        will get its own process.

        Args:
            transformers: list of transformers of length equal to X.shape[1] or
                len(cols).
            cols: DataFrame.columns, list of columns. See description for
                when to pass.
            criterion: column grouping criterion.

        Returns:
            A dict where each entry can be used to separately access an
            individual column from X.

        Raises:
            ValueError: input does not matched defined format.

        """
        if len(transformers) != len(cols):
            raise ValueError(
                "number of transformer steps: '{}' "
                "does not match number of "
                "column groups: '{}'".format(len(transformers), len(cols))
            )
        pm = PreparerMapping()
        for i, group_col in enumerate(cols):
            group_col = (
                [group_col]
                if not isinstance(group_col, (list, tuple))
                else group_col
            )
            pm.add(
                group_col,
                transformers[i],
                (criterion[i] if criterion is not None else i),
            )
        return pm

    @classmethod
    def logging_name(cls):
        """Return the logging name for this transformer.

        Returns:
            See return.

        """
        return "DataPreparerStep: {} ".format(cls.__name__)

    def parallelize_mapping(self, column_mapping):
        """Create parallelized workflow for column_mapping.

        Each group of cols that is separated has the key: 'group_name' and a
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
        # we will first map all groups that have no
        # interdependencies with other groups. Then, we will do all the rest
        # of the groups after as they will be performed step-by-step
        # parallelized.
        for group_name, group_process in column_mapping.items():

            transformer_list = _check_parallelizable_batch(
                group_process, group_name
            )
            if transformer_list is not None:
                final_mapping[group_name] = transformer_list
        # if len(final_mapping) < len(column_mapping):  # then there
        #     # must be groups of columns that have interdependcies.
        #     # CURRENTLy DISABLED.
        #     steps, all_cols = _batch_parallelize(column_mapping)
        #     final_mapping["grouped_pipeline"] = [
        #         "grouped_pipeline",
        #         Pipeline(steps),
        #         all_cols,
        #     ]

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
        return ParallelProcessor(
            group_transformer_list,
            n_jobs=config.get_n_jobs_config(),
            collapse_index=True,
        )

    def get_mapping(self, X):
        """Return a PreparerMapping object.

        The return has 2 major components:
            1: the number of parallel operations to be performed on
            the DataFrame.

            For each parallel operation, there is:
            2: the number of steps/operations. This can be viewed as
            groups of columns being passed to a single smart transformer. For
            instance, you may pass a single column each to its on smart
            transformer (say, to clean each column individually), or all
            columns to a single smart transformer (for instance,
            for dimensionality reduction).

        To have a None step, pass in [None].

        This data structure is constructed by using PreparerMapping, to
        more easily align with user configuration and serialization. The outer
        levels defines the number of groups of operations. The second layer is
        a namedtuple with two keys: 'inputs' and 'steps'. There are two
        accepted formats: inputs is a nested tuple of length 1, or nested
        tuple where the number of nested tuples is equal to the number of
        steps. In this case, each step is passed the inputs defined in each
        tuple. This latter case is not yet fully implemented.

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
        self.fit_transform(X, *args, **kwargs)
        return self

    def check_process(self, X):
        """If fit was never called, makes sure to create the parallel process.

        Args:
            X: input DataFrame

        """
        logging.debug(
            "DataPreparerStep: {} called check_process".format(
                self.__class__.__name__
            )
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
        try:
            return self._parallel_process.fit_transform(X, y=y, **fit_params)
        except AttributeError:
            if getattr(self, "_parallel_process", None) is None:
                self.check_process(X)
        except KeyError as e:
            if str(e).find("not in index") != -1:
                # This indicates that a transformation step was changed and
                # now does not correctly reflect the generated DataFrame as
                # this step. We will thus reinitialize the _parallel_process
                # so that the best pipeline for this step will be found.
                self.check_process(X)
        finally:
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

        Raises:
            ValueError: if not fitted.

        """
        if getattr(self, "_parallel_process", None) is None:
            raise ValueError("not fitted.")
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

    @classmethod
    def _get_param_names(cls):
        """Get iteratively __init__ params for all classes until PreparerStep.

        Overridden to add this parent classes' params to children and to
        include _parallel_process. _get_param_names holds the logic for
        getting all parent params.

        This method is implemented as a convenience for any child. It will
        automatically climb the MRO for a child until it reaches this class
        (the last parent who's __init__ params we care about). Also adds
        _parallel_process to the sklearn get_params API.

        Returns:
            params for all parents up to and including PreparerStep.
            Includes the calling classes params.

        """
        params = super()._get_param_names()
        while cls.__name__ != PreparerStep.__name__:
            cls = cls.__mro__[1]
            params += cls._get_param_names()
        if "_parallel_process" not in params:
            params += ["_parallel_process"]
        return params
