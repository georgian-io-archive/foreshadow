"""Data preparation and foreshadow pipeline."""

from sklearn.pipeline import Pipeline

from foreshadow.serializers import (
    PipelineSerializerMixin,
    _make_deserializable,
    _make_serializable,
)
from foreshadow.steps import (
    CleanerMapper,
    FeatureEngineererMapper,
    FeatureReducerMapper,
    FeatureSummarizerMapper,
    IntentMapper,
    Preprocessor,
)
from foreshadow.utils import ConfigureColumnSharerMixin

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


class DataPreparer(
    Pipeline, PipelineSerializerMixin, ConfigureColumnSharerMixin
):
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
        column_sharer=None,
        cleaner_kwargs=None,
        intent_kwargs=None,
        summarizer_kwargs=None,
        engineerer_kwargs=None,
        preprocessor_kwargs=None,
        reducer_kwargs=None,
        y_var=None,
        **kwargs
    ):
        cleaner_kwargs_ = _none_to_dict(
            "cleaner_kwargs", cleaner_kwargs, column_sharer
        )
        intent_kwargs_ = _none_to_dict(
            "intent_kwargs", intent_kwargs, column_sharer
        )
        summarizer_kwargs_ = _none_to_dict(
            "summarizer_kwargs", summarizer_kwargs, column_sharer
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
                ("intent", IntentMapper(**intent_kwargs_)),
                (
                    "feature_summarizer",
                    FeatureSummarizerMapper(**summarizer_kwargs_),
                ),
                (
                    "feature_engineerer",
                    FeatureEngineererMapper(**engineerer_kwargs_),
                ),
                ("feature_preprocessor", Preprocessor(**preprocessor_kwargs_)),
                ("feature_reducer", FeatureReducerMapper(**reducer_kwargs_)),
            ]
        else:
            steps = [("output", NoTransform())]
        if "steps" in kwargs:  # needed for sklearn estimator clone,
            # which will try to init the object using get_params.
            steps = kwargs.pop("steps")

        self.column_sharer = column_sharer
        self.y_var = y_var
        super().__init__(steps, **kwargs)

    def _get_params(self, attr, deep=True):
        # attr will be 'steps' if called from pipeline.get_params()
        out = super()._get_params(attr, deep)
        steps = getattr(self, attr)
        out.update({"steps": steps})  # manually
        # adding steps to the get_params()
        return out

    def dict_serialize(self, deep=False):
        """Serialize the data preparer.

        Args:
            deep: see super.

        Returns:
            dict: serialized data preparer.

        """
        params = self.get_params(deep=False)
        serialized = _make_serializable(
            params, serialize_args=self.serialize_params
        )
        column_sharer_serialized = serialized.pop("column_sharer", None)
        serialized = self.__remove_key_from(serialized, target="column_sharer")
        # Add back the column_sharer in the end only once.
        serialized["column_sharer"] = column_sharer_serialized
        steps = serialized["steps"]
        steps_reformatted = [{step[0]: step[1]} for step in steps]
        serialized["steps"] = steps_reformatted
        return serialized

    @classmethod
    def dict_deserialize(cls, data):
        """Deserialize the data preparer.

        Args:
            data: serialized data preparer in JSON format.

        Returns:
            a reconstructed data preparer.

        """
        params = _make_deserializable(data)
        params["steps"] = [list(step.items())[0] for step in params["steps"]]
        deserialized = cls(**params)

        deserialized.configure_column_sharer(deserialized.column_sharer)

        return deserialized

    def configure_column_sharer(self, column_sharer):
        """Configure column sharer for all the underlying components recursively.

        Args:
            column_sharer: the column sharer instance.

        """
        for step in self.steps:
            if hasattr(step[1], "configure_column_sharer"):
                step[1].configure_column_sharer(column_sharer)

    def __remove_key_from(self, data, target="column_sharer"):
        """Remove all column sharer block recursively from serialized data preparer.

        Only the column sharer in the data preparer is preserved.

        Args:
            data: serialized data preparer (raw)
            target: string that should match as a suffix of a key

        Returns:
            dict: a cleaned up serialized data preparer

        """
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
