"""SmartResolver for ResolverMapper step."""

from foreshadow.intents import Categorical, Droppable, Numeric, Text
from foreshadow.logging import logging
from foreshadow.smart.intent_resolving.core import (
    IntentResolver as AutoIntentResolver,
)
from foreshadow.smart.smart import SmartTransformer
from foreshadow.utils import (
    AcceptedKey,
    DataSamplingMixin,
    Override,
    get_transformer,
)


_temporary_naming_conversion = {
    "Numerical": Numeric.__name__,
    "Categorical": Categorical.__name__,
    "Droppable": Droppable.__name__,
    "Text or Datetime": Text.__name__,
}


def _temporary_naming_convert(auto_ml_intent_name):
    if auto_ml_intent_name in _temporary_naming_conversion:
        return _temporary_naming_conversion[auto_ml_intent_name]
    else:
        raise KeyError(
            "No such intent type {} exists.".format(auto_ml_intent_name)
        )


class IntentResolver(SmartTransformer, DataSamplingMixin):
    """Determine the intent for a particular column.

    Params:
        **kwargs: kwargs to pass to individual intent constructors

    """

    validate_wrapped = False

    def __init__(self, column=None, **kwargs):
        super().__init__(**kwargs)
        self.column = column
        self.column_intent = None
        # self.cache_manager = cache_manager

    def _resolve_intent(self, X, y=None):
        """Pick the intent with the highest confidence score.

        Args:
            X: the data frame to be processed.
            y: None

        Returns:
            The intent class that best matches the input data.

        """
        X_sampled = self.sample_data_frame(df=X)
        auto_intent_resolver = AutoIntentResolver(X_sampled)
        intent_pd_series = auto_intent_resolver.predict()
        return _temporary_naming_convert(intent_pd_series[[0]].values[0])

    def resolve(self, X, *args, **kwargs):
        """Pick the appropriate transformer if necessary.

        Note:
            Column info sharer is set based on the chosen transformer.

        Args:
            X: input observations
            *args: args to pass to resolve
            **kwargs: params to resolve

        """
        # Override the SmartTransformer resolve method to allow the setting of
        # column info sharer data when resolving.
        super().resolve(X, *args, **kwargs)
        column_name = self.column
        self.column_intent = self.transformer.__class__.__name__
        logging.info(
            "Column {} has intent type: {}".format(
                column_name, self.column_intent
            )
        )

    def pick_transformer(self, X, y=None, **fit_params):
        """Get best intent transformer for a given column.

        Note:
            This function also sets the cache_manager

        Args:
            X: input DataFrame
            y: input labels
            **fit_params: fit_params

        Returns:
            Best intent transformer.

        """
        column = X.columns[0]
        override_key = "_".join([Override.INTENT, column])
        if override_key in self.cache_manager[AcceptedKey.OVERRIDE]:
            intent_override = self.cache_manager[AcceptedKey.OVERRIDE][
                override_key
            ]
            intent_class = get_transformer(intent_override)
        else:
            intent_class = get_transformer(self._resolve_intent(X, y=y))

        return intent_class()
