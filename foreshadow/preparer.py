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
        **kwargs,
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
            ]
        else:
            steps = [("output", NoTransform())]
        if 'steps' in kwargs:  # needed for sklearn estimator clone,
            # which will try to init the object using get_params.
            steps = kwargs.pop('steps')

        self.column_sharer = column_sharer
        self.y_var = y_var
        # modeler_kwargs_ = _none_to_dict(
        #     "modeler_kwargs", modeler_kwargs, column_sharer
        # )
        super().__init__(steps, **kwargs)

    def _get_params(self, attr, deep=True):
        # attr will be 'steps' if called from pipeline.get_params()
        out = super()._get_params(attr, deep)
        steps = getattr(self, attr)
        out.update({"steps": steps})  # manually
        # adding steps to the get_params()
        return out
