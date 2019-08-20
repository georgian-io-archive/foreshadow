"""Data preparation and foreshadow pipeline."""
# flake8: noqa

import inspect

from sklearn.pipeline import Pipeline

from foreshadow.serializers import PipelineSerializerMixin, _make_serializable
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
        **kwargs
    ):
        cleaner_kwargs_ = _none_to_dict(
            "cleaner_kwargs", cleaner_kwargs, column_sharer
        )
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
        if not y_var:
            steps = [
                ("data_cleaner", CleanerMapper(**cleaner_kwargs_)),
                # ("intent", IntentMapper(**intent_kwargs_)),
                # (
                #     "feature_engineerer",
                #     FeatureEngineererMapper(**engineerer_kwargs_),
                # ),
                # ("feature_preprocessor", Preprocessor(**preprocessor_kwargs_)),
                # ("feature_reducer", FeatureReducerMapper(**reducer_kwargs_)),
            ]
        else:
            steps = [("output", NoTransform())]
        if "steps" in kwargs:  # needed for sklearn estimator clone,
            # which will try to init the object using get_params.
            steps = kwargs.pop("steps")

        self.column_sharer = column_sharer
        self.y_var = y_var
        # modeler_kwargs_ = _none_to_dict(
        #     "modeler_kwargs", modeler_kwargs, column_sharer
        # )
        super().__init__(steps, **kwargs)

    # def dict_serialize(self, deep=True):
    #     serialized = super().dict_serialize(deep=deep)
    #     import pdb;pdb.set_trace()
    #     steps = serialized["steps"]
    #     steps_reformatted = [{step[0]: step[1]} for step in steps]
    #     serialized["steps"] = steps_reformatted
    #     return serialized

    def _get_params(self, attr, deep=True):
        # attr will be 'steps' if called from pipeline.get_params()
        out = super()._get_params(attr, deep)
        steps = getattr(self, attr)
        out.update({"steps": steps})  # manually
        # adding steps to the get_params()
        return out

    # def dict_serialize(self, deep=False):
    #     """Serialize the init parameters (dictionary form) of a pipeline.
    #
    #     This method removes redundant column_sharers in the individual
    #     steps.
    #
    #     Note:
    #         This recursively serializes the individual steps to facilitate a
    #         human readable form.
    #
    #     Args:
    #         deep (bool): If True, will return the parameters for this estimator
    #             recursively
    #
    #     Returns:
    #         dict: The initialization parameters of the pipeline.
    #
    #     """
    #     serialized = super().dict_serialize(deep=deep)
    #     column_sharer_serialized = serialized.pop("column_sharer", None)
    #     # Remove all instance of column_sharer from the serialized recursively.
    #     serialized = self.__remove_key_from(serialized, target="column_sharer")
    #     # Add back the column_sharer in the end only once.
    #     serialized["column_sharer"] = column_sharer_serialized
    #     return serialized

    def dict_serialize(self, deep=True):
        import pdb

        pdb.set_trace()
        params = self.get_params(deep=deep)
        selected_params = self.__create_selected_params(params)
        serialized = _make_serializable(
            selected_params, serialize_args=self.serialize_params
        )
        column_sharer_serialized = serialized.pop("column_sharer", None)
        serialized = self.__remove_key_from(serialized, target="column_sharer")
        # Add back the column_sharer in the end only once.
        serialized["column_sharer"] = column_sharer_serialized
        steps = serialized["steps"]
        steps_reformatted = [{step[0]: step[1]} for step in steps]
        serialized["steps"] = steps_reformatted
        return serialized

    def __create_selected_params(self, params):
        init_params = inspect.signature(self.__init__).parameters
        selected_params = {
            name: params.pop(name)
            for name in init_params
            if name not in ["self", "kwargs"]
        }
        selected_params["steps"] = params.pop("steps")
        return selected_params

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


if __name__ == "__main__":
    pass
