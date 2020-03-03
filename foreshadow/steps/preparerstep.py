"""General base classes used across Foreshadow."""

from foreshadow.base import BaseEstimator, TransformerMixin
from foreshadow.utils.common import ConfigureCacheManagerMixin

from ..cachemanager import CacheManager


class PreparerStep(
    BaseEstimator, TransformerMixin, ConfigureCacheManagerMixin
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

    def __init__(self, cache_manager=None, **kwargs):  # noqa
        """Set the original pipeline steps internally.

        Takes a list of desired SmartTransformer steps and stores them as
        self._steps. Constructs self an sklearn pipeline object.

        Args:
            cache_manager: ColumnSharer instance to be shared across all steps.
            **kwargs: kwargs to PIpeline constructor.

        """
        self.feature_processor = None
        self.cache_manager = cache_manager
        if self.cache_manager is None:
            self.cache_manager = CacheManager()
        super().__init__(**kwargs)

    def has_fitted(self):
        """Check if the prepare step has been fitted.

        Returns:
            Whether the step has been fitted.

        """
        return self.feature_processor is not None

    @classmethod
    def logging_name(cls):
        """Return the logging name for this transformer.

        Returns:
            See return.

        """
        return "DataPreparerStep: {} ".format(cls.__name__)

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
        pass
        # logging.debug(
        #     "DataPreparerStep: {} called check_process".format(
        #         self.__class__.__name__
        #     )
        # )
        # default_parallel_process = self.parallelize_smart_steps(X)
        # if self._parallel_process is None:
        #     self._parallel_process = default_parallel_process
        # else:
        #     self._handle_intent_override(default_parallel_process)

        # self._parallel_process = self.parallelize_smart_steps(X)

    def _handle_intent_override(self, default_parallel_process):
        """Handle intent override and see override in the child classes.

        Different preparestep may handle the intent override differently but in
        general it involves checking if the column groups have changed and need
        to reset to the default value. TODO it may be beneficial to keep track
        of both the old and new intents of columns as it may help the update of
        groups of multiple columns.

        Args:
            default_parallel_process: the default_parallel_process

        """
        pass

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
        if getattr(self, "feature_processor", None) is None:
            raise ValueError("not fitted.")
        return self.feature_processor.transform(X, *args, **kwargs)

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
        return self.feature_processor.inverse_transform(X, *args, **kwargs)

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
