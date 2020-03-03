"""Data preparation and foreshadow pipeline."""

from sklearn.pipeline import Pipeline

from foreshadow.smart import CategoricalEncoder
from foreshadow.steps import (
    CleanerMapper,
    DataExporterMapper,
    FeatureSummarizerMapper,
    FlattenMapper,
    IntentMapper,
    Preprocessor,
)
from foreshadow.utils import ConfigureCacheManagerMixin, ProblemType

from .concrete import NoTransform


def _none_to_dict(name, val, cache_manager=None):
    """Transform input kwarg to valid dict, handling sentinel value.

    Accepts a single kwarg.

    Args:
        name: the kwarg name
        val: the kwarg value to ensure is proper format for kwargs.
        cache_manager: if None, do nothing. If a value, add to kwarg values.

    Returns:
        kwarg set to default

    Raises:
        ValueError: if value of kwarg is not a valid value for kwarg (dict,
            None). Also if > 1 kwargs passed.

    """
    val = {} if val is None else val
    if not isinstance(val, dict):
        raise ValueError(
            "value for kwarg: {} must be dict or " "None.".format(name)
        )
    if cache_manager is not None:
        val["cache_manager"] = cache_manager
    return val


class DataPreparer(Pipeline, ConfigureCacheManagerMixin):
    """Predefined pipeline for the foreshadow workflow.

    1. Cleaning
    2. Intent selection (data type, one of Categorical, Numerical, and Text)
    3. Engineering (Based on intent. Feature generation and reduction)
    4. Preprocessing (Based on intent. Scaling, one hot encoding, etc.)
    5. Reducing (loosely based on intent. Dimensionality reduction).

    In customizing any of the components within these steps:
        concrete transformers, SmartTransformers, their params, etc.,
    the produced columns may change. This entire workflow uses column
    names to assign steps to their associated columns, so, changing
    components of this workflow may change the column names in the case
    that column names were generated for your column based on the
    processing step. In this event, if the we will reinstantiate the
    entire step (cleaner, intent, etc.) for the column only when necessary.
    """

    # TODO In the future, we will attempt to make this smarter by only
    #  modifiying the specific transformers needed within each step.
    def __init__(
        self,
        cache_manager=None,
        flattener_kwargs=None,
        cleaner_kwargs=None,
        intent_kwargs=None,
        summarizer_kwargs=None,
        engineerer_kwargs=None,
        preprocessor_kwargs=None,
        reducer_kwargs=None,
        exporter_kwargs=None,
        problem_type=None,
        y_var=None,
        **kwargs
    ):
        flattener_kwargs = _none_to_dict(
            "flattener_kwargs", flattener_kwargs, cache_manager
        )
        cleaner_kwargs_ = _none_to_dict(
            "cleaner_kwargs", cleaner_kwargs, cache_manager
        )
        intent_kwargs_ = _none_to_dict(
            "intent_kwargs", intent_kwargs, cache_manager
        )
        summarizer_kwargs_ = _none_to_dict(
            "summarizer_kwargs", summarizer_kwargs, cache_manager
        )
        # # engineerer_kwargs_ = _none_to_dict(
        # #     "engineerer_kwargs", engineerer_kwargs, cache_manager
        # # )
        preprocessor_kwargs_ = _none_to_dict(
            "preprocessor_kwargs", preprocessor_kwargs, cache_manager
        )
        # # reducer_kwargs_ = _none_to_dict(
        # #     "reducer_kwargs", reducer_kwargs, cache_manager
        # # )
        exporter_kwargs_ = _none_to_dict(
            "exporter_kwargs", exporter_kwargs, cache_manager
        )
        if not y_var:
            steps = [
                ("data_flattener", FlattenMapper(**flattener_kwargs)),
                ("data_cleaner", CleanerMapper(**cleaner_kwargs_)),
                ("intent", IntentMapper(**intent_kwargs_)),
                (
                    "feature_summarizer",
                    FeatureSummarizerMapper(**summarizer_kwargs_),
                ),
                # (
                #     "feature_engineerer",
                #     FeatureEngineererMapper(**engineerer_kwargs_),
                # ),
                ("feature_preprocessor", Preprocessor(**preprocessor_kwargs_)),
                # ("feature_reducer", FeatureReducerMapper(**reducer_kwargs_)),
                ("feature_exporter", DataExporterMapper(**exporter_kwargs_)),
            ]
        else:
            if problem_type == ProblemType.REGRESSION:
                steps = [("output", NoTransform())]
            elif problem_type == ProblemType.CLASSIFICATION:
                steps = [
                    (
                        "output",
                        CategoricalEncoder(
                            y_var=True, cache_manager=cache_manager
                        ),
                    )
                ]
            else:
                raise ValueError(
                    "Invalid Problem " "Type {}".format(problem_type)
                )
        if "steps" in kwargs:  # needed for sklearn estimator clone,
            # which will try to init the object using get_params.
            steps = kwargs.pop("steps")

        self.cache_manager = cache_manager
        self.y_var = y_var
        self.problem_type = problem_type
        super().__init__(steps, **kwargs)

    def _get_params(self, attr, deep=True):
        # attr will be 'steps' if called from pipeline.get_params()
        out = super()._get_params(attr, deep)
        steps = getattr(self, attr)
        out.update({"steps": steps})  # manually
        # adding steps to the get_params()
        return out

    # TODO Remove this code if we decided to not include the JSON serialization
    # def dict_serialize(self, deep=False):
    #     """Serialize the data preparer.
    #
    #     Args:
    #         deep: see super.
    #
    #     Returns:
    #         dict: serialized data preparer.
    #
    #     """
    #     params = self.get_params(deep=False)
    #     serialized = _make_serializable(
    #         params, serialize_args=self.serialize_params
    #     )
    #     cache_manager_serialized = serialized.pop("cache_manager", None)
    #     serialized = self.__remove_key_from(serialized,
    #                                         target="cache_manager")
    #     # Add back the cache_manager in the end only once.
    #     serialized["cache_manager"] = cache_manager_serialized
    #     steps = serialized["steps"]
    #     steps_reformatted = [{step[0]: step[1]} for step in steps]
    #     serialized["steps"] = steps_reformatted
    #     return serialized
    #
    # @classmethod
    # def dict_deserialize(cls, data):
    #     """Deserialize the data preparer.
    #
    #     Args:
    #         data: serialized data preparer in JSON format.
    #
    #     Returns:
    #         a reconstructed data preparer.
    #
    #     """
    #     params = _make_deserializable(data)
    #     params["steps"] = [list(step.items())[0] for step in params["steps"]]
    #     deserialized = cls(**params)
    #
    #     deserialized.configure_cache_manager(deserialized.cache_manager)
    #
    #     return deserialized
    #
    # def configure_cache_manager(self, cache_manager):
    #     """Configure cache_manager for all the underlying components
    #     recursively.
    #
    #     Args:
    #         cache_manager: the cache_manager instance.
    #
    #     """
    #     for step in self.steps:
    #         if self.y_var:
    #             step[1].cache_manager = cache_manager
    #         elif hasattr(step[1], "configure_cache_manager"):
    #             step[1].configure_cache_manager(cache_manager)
    #
    # def __remove_key_from(self, data, target="cache_manager"):
    #     """Remove all cache_manager block recursively from serialized data
    #     preparer.
    #
    #     Only the cache_manager in the data preparer is preserved.
    #
    #     Args:
    #         data: serialized data preparer (raw)
    #         target: string that should match as a suffix of a key
    #
    #     Returns:
    #         dict: a cleaned up serialized data preparer
    #
    #     """
    #     if isinstance(data, dict):
    #         matching_keys = [key for key in data if key.endswith(target)]
    #         for mk in matching_keys:
    #             del data[mk]
    #         data = {
    #             key: self.__remove_key_from(data[key], target=target)
    #             for key in data
    #         }
    #     elif isinstance(data, list):
    #         data = [
    #             self.__remove_key_from(item, target=target) for item in data
    #         ]
    #     return data


#
#
# if __name__ == "__main__":
#     pass
