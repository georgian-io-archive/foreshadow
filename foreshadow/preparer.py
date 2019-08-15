"""Data preparation and foreshadow pipeline."""

from sklearn.pipeline import Pipeline

from foreshadow.pipeline import PipelineSerializerMixin
from foreshadow.steps import (
    CleanerMapper,
    FeatureEngineererMapper,
    FeatureReducerMapper,
    IntentMapper,
    Preprocessor,
)

from .concrete import NoTransform


def _none_to_dict(name, val, column_sharer=None):
    """Transform input kwarg to valid dict, handling sentinel value.

    Accepts a single kwarg.

    Args:
        name: the kwarg name
        val: the kwarg value to ensure is proper format for kwargs.
        column_sharer: if None, do nothing. If a value, add to kwarg values.

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
    if column_sharer is not None:
        val["column_sharer"] = column_sharer
    return val


class DataPreparer(Pipeline, PipelineSerializerMixin):
    """Predefined pipeline for the foreshadow workflow."""

    def __init__(
        self,
        column_sharer=None,
        cleaner_kwargs=None,
        intent_kwargs=None,
        engineerer_kwargs=None,
        preprocessor_kwargs=None,
        reducer_kwargs=None,
        modeler_kwargs=None,
        y_var=None,
    ):
        self.column_sharer = column_sharer
        # TODO look at fixing structure so we don't have to import inside init.
        cleaner_kwargs_ = _none_to_dict(
            "cleaner_kwargs", cleaner_kwargs, column_sharer
        )
        self.y_var = y_var
        intent_kwargs_ = _none_to_dict(
            "intent_kwargs", intent_kwargs, column_sharer
        )
        engineerer_kwargs_ = _none_to_dict(
            "engineerer_kwargs", engineerer_kwargs, column_sharer
        )
        preprocessor_kwargs_ = _none_to_dict(
            "preprocessor_kwargs", preprocessor_kwargs, column_sharer
        )
        reducer_kwargs_ = _none_to_dict(
            "reducer_kwargs", reducer_kwargs, column_sharer
        )
        # modeler_kwargs_ = _none_to_dict(
        #     "modeler_kwargs", modeler_kwargs, column_sharer
        # )
        if not self.y_var:
            super().__init__(
                steps=[
                    ("data_cleaner", CleanerMapper(**cleaner_kwargs_)),
                    ("intent", IntentMapper(**intent_kwargs_)),
                    (
                        "feature_engineerer",
                        FeatureEngineererMapper(**engineerer_kwargs_),
                    ),
                    (
                        "feature_preprocessor",
                        Preprocessor(**preprocessor_kwargs_),
                    ),
                    (
                        "feature_reducer",
                        FeatureReducerMapper(**reducer_kwargs_),
                    ),
                    # ('model_selector', modeler_kwargs_)
                ]  # TODO add each of these components
            )
        else:
            super().__init__(steps=[("output", NoTransform())])

    def _get_params(self, attr, deep=True):
        # attr will be 'steps' if called from pipeline.get_params()
        out = super()._get_params(attr, deep)
        steps = getattr(self, attr)
        out.update({"steps": steps})  # manually
        # adding steps to the get_params()
        return out

    def dict_serialize(self, deep=False):
        """Serialize the init parameters (dictionary form) of a pipeline.

        This method removes redundant column_sharers in the individual
        steps.

        Note:
            This recursively serializes the individual steps to facilitate a
            human readable form.

        Args:
            deep (bool): If True, will return the parameters for this estimator
                recursively

        Returns:
            dict: The initialization parameters of the pipeline.

        """
        serialized = super().dict_serialize(deep=deep)
        column_sharer_serialized = serialized.pop("column_sharer", None)
        # Remove all instance of column_sharer from the serialized recursively.
        serialized = self.__remove_key_from(serialized, target="column_sharer")
        # Add back the column_sharer in the end only once.
        serialized["column_sharer"] = column_sharer_serialized
        return serialized

    def __remove_key_from(self, data, target="column_sharer"):
        if isinstance(data, dict):
            matching_keys = [key for key in data if key.endswith(target)]
            for mk in matching_keys:
                del data[mk]
            data = {
                key: self.__remove_key_from(data[key], target=target)
                for key in data
            }
        elif isinstance(data, list):
            data = [
                self.__remove_key_from(item, target=target) for item in data
            ]
        return data
