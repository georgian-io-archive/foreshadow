"""Data preparation and foreshadow pipeline."""
from sklearn.pipeline import Pipeline

from foreshadow.cleaners.data_cleaner import DataCleaner as _DataCleaner


def _none_to_dict(**kwargs):
    """Transform input kwarg to valid dict, handling sentinel value.

    Accepts a single kwarg by default.

    Args:
        **kwargs: the kwarg value to ensure is proper format for kwargs.

    Returns:
        kwarg set to default

    Raises:
        ValueError: if value of kwarg is not a valid value for kwarg (dict,
            None). Also if > 1 kwargs passed.

    """
    if len(kwargs) > 1:
        raise ValueError("Pass only 1 kwarg at a time.")
    for key, val in kwargs.items():
        val = {} if val is None else val
        if not isinstance(val, dict):
            raise ValueError(
                "value for kwarg: {} must be dict or " "None.".format(key)
            )
        return val


class DataPreparer(Pipeline):
    """Predefined pipeline for the foreshadow workflow."""

    def __init__(
        self,
        cleaner_kwargs=None,
        intent_kwargs=None,
        engineerer_kwargs=None,
        preprocessor_kwargs=None,
        reducer_kwargs=None,
        modeler_kwargs=None,
    ):
        cleaner_kwargs_ = _none_to_dict(cleaner_kwargs=cleaner_kwargs)
        # intent_kwargs_ = _none_to_dict(intent_kwargs=intent_kwargs)
        # engineerer_kwargs_ = _none_to_dict(
        # engineerer_kwargs=engineerer_kwargs
        # )
        # preprocessor_kwargs_ = _none_to_dict(
        #     preprocessor_kwargs=preprocessor_kwargs
        # )
        # reducer_kwargs_ = _none_to_dict(reducer_kwargs=reducer_kwargs)
        # modeler_kwargs_ = _none_to_dict(modeler_kwargs=modeler_kwargs)

        super().__init__(
            steps=[
                ("data_cleaner", _DataCleaner(**cleaner_kwargs_)),
                # ('intent', intent_kwargs_),
                # ('feature_engineerer', engineerer_kwargs_),
                # ('feature_preprocessor', preprocessor_kwargs_),
                # ('feature_reducer', reducer_kwargs_,),
                # ('model_selector', modeler_kwargs_)
            ]  # TODO add each of these components
        )
