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
    """Predefined pipeline for foreshadow workflow. This Pipeline has 5 steps.

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

    def check_columnsharer_sync(self):
        """Ensure that all column_sharer instances remain in sync."""
        # TODO implement this. It should be called before predict or fit.
        #  Leverage Jing's code in the new branch to check all the
        #  get/setparams of each child and if they have a columnsharer,
        #  delete their instance and instead set it with this one.
        pass

    def fit(self, *args, **kwargs):
        """Fit the sklearn pipeline and ensure columnsharer's are shared.

        Uses self.check_columnsharer_sync() to ensure that the columnsharer
        object is kept in sync with all subobjects in the nested pipelines.
        This may be an issue when this pipeline is cloned, as the
        initialization through sklearn's internal clone may not yet play
        nice with our internal init synchronization protocol.

        Args:
            *args: See super.
            **kwargs: See super.

        Returns:
            See super.

        """
        self.check_columnsharer_sync()
        return super().fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """Predict using sklearn pipeline and ensure columnsharer's are shared.

        Uses self.check_columnsharer_sync() to ensure that the columnsharer
        object is kept in sync with all subobjects in the nested pipelines.
        This may be an issue when this pipeline is cloned, as the
        initialization through sklearn's internal clone may not yet play
        nice with our internal init synchronization protocol.

        Args:
            *args: See super.
            **kwargs: See super.

        Returns:
            See super.

        """
        self.check_columnsharer_sync()
        return super().fit(*args, **kwargs)
