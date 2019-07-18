"""General base classes used across Foreshadow."""
from abc import abstractmethod
from copy import deepcopy

from sklearn.pipeline import Pipeline

from foreshadow.transformers.base import ParallelProcessor, SmartTransformer
from foreshadow.utils import check_df


class PreparerStep(Pipeline):
    """Abstract Base class for any pipeline step of DataPreparer.

    This class automatically wraps the defined pipeline to make it
    parallelizable across a DataFrame. To make this possible and still be
    customizable, subclasses must implement get_transformer_list and
    get_transformer_weights, which are two inputs the ParallelProcessor
    (the class used to make parallelization possible).


    The transformer_list represents the mapping from columns to
    transformers, in the form of ['name', 'transformer', ['cols']],
    where the [cols] are the cols for transformer 'transformer. These
    transformers should be SmartTransformers for any subclass.

    The transformer_weights are multiplicative weights for features per
    transformer. Keys are transformer names, values the weights.

    """

    def __init__(self, steps, *args, **kwargs):
        """Set the original pipeline steps internally.

        Takes a list of desired SmartTransformer steps and stores them as
        self._steps. Constructs self an sklearn pipeline object.

        Args:
            steps: list of ('name', 'SmartTransformer') tuples, where the
                latter is a class referenece.
            *args: args to Pipeline constructor.
            **kwargs: kwargs to PIpeline constructor.
        """
        self._steps = steps  # class pointers
        super().__init__(steps, *args, **kwargs)

    @staticmethod
    def one_smart_all_cols(smart, X):
        """Apply separate smart transformer to each column.

        Args:
            smart: the SmartTransformer to apply.
            X: the input dataset points

        Returns:
            (transformer_list, transformer_weights) as inputs for
            ParallelProcessor

        """
        transformer_list = [(smart.__name__, smart(), [col]) for col in X]
        transformer_weights = None
        return transformer_list, transformer_weights

    def parallelize_smart_steps(self, X):
        """Make self.steps for internal pipeline methods parallelized.

        Takes self._steps passed at construction time and wraps each step
        with ParallelProcessor to parallelize it across a DataFrame's columns.
        Made possible sing get_transformer_list and get_transformer_weights
        which must be implemented by the subclass.

        Args:
            X: DataFrame

        """
        steps = []
        for step in self._steps:
            name = step[0]
            transformer = step[1]
            transformer_list = self.get_transformer_list(transformer, X)
            transformer_weights = self.get_transformer_weights(transformer, X)
            steps.append(
                (
                    name,
                    ParallelProcessor(
                        transformer_list,
                        transformer_weights=transformer_weights,
                    ),
                )
            )
        self.steps = steps

    @abstractmethod
    def get_transformer_list(self, transformer, X):
        """Return valid transformer_list for ParallelProcessor.

        Args:
            transformer: SmartTransformer to use
            X: input DataFrame

        Returns:
            ['name', 'transformer', ['cols']] format where each column
            should be included exactly once.

        .. # noqa: I202

        """
        pass

    @abstractmethod
    def get_transformer_weights(self, transformer, X):
        """Return valid transformer_weights for ParallelProcessor.

        Args:
            transformer: SmartTransformer to use
            X: input DataFrame

        Returns:
            {'transformer': weight} weighting. See FeatureUnion object.

        .. # noqa: I202

        """
        pass

    # main underlying fit(_transform)
    # method called for fit, fit_transform, etc. Returns fit_transformed X.
    def _fit(self, X, *args, **kwargs):
        """Parallelize workflow then fit_transform data.

        Standard sklearn Pipeline fit where each step in the pipeline is
        parallelized using self.parallelize_smart_steps.

        Args:
            X: input DataFrame
            *args: args to _fit
            **kwargs: kwargs to _fit

        Returns:
            transformed data handled by Pipeline._fit

        """
        self.parallelize_smart_steps(X)
        return super()._fit(X, *args, **kwargs)  # internally will iterate
        # over steps in _fit.


class SinglePreparerStep(PreparerStep):
    def __init__(self, smart_transformer):
        smart_transformer = [smart_transformer]
        super().__init__(smart_transformer)
        self.transformer = deepcopy(self._steps[0])
        del self._steps

    def _fit(self, *args, **kwargs):
        self.steps = (self.transformer.__name__, self.transformer)
        super().fit(*args, **kwargs)
        del self.steps

    def get_transformer_list(self, transformer, X):
        """Return transformer mapping to columns.

        Args:
            transformer: SmartTransformer to use
            X: input DataSet

        Returns:
            transformer: col for each column mapping.

        """
        return self.one_smart_all_cols(transformer, X)[0]  # create a
        # separate SmartCleaner for each column

    def get_transformer_weights(self, transformer, X):
        """Return no transformer_weights.

        Args:
            transformer: SmartTransformer to use
            X: input DataSet

        Returns:
            None

        """
        return self.one_smart_all_cols(transformer, X)[1]  # don't use any
        # weighting.
