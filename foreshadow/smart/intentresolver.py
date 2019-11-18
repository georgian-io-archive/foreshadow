"""SmartResolver for ResolverMapper step."""

from foreshadow.config import config
from foreshadow.smart.smart import SmartTransformer
from foreshadow.utils import Override, get_transformer


class IntentResolver(SmartTransformer):
    """Determine the intent for a particular column.

    Params:
        **kwargs: kwargs to pass to individual intent constructors

    """

    validate_wrapped = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _resolve_intent(self, X, y=None):  # noqa
        """Pick the intent with the highest confidence score.

        Note:
            In the case of ties, the intent `defined first \
            <https://docs.python.org/3/library/functions.html#max>`_ in the
            config list is chosen, the priority order is defined by the config
            file `resolver` section.

        Args:
            X: input observations
            y: not used

        Returns:
            The intent class that best matches the input data.

        .. # noqa: S001

        """
        intent_list = config.get_intents()
        return max(intent_list, key=lambda intent: intent.get_confidence(X))

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
        column_name = X.columns[0]
        self.cache_manager[
            "intent", column_name
        ] = self.transformer.__class__.__name__

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
        if override_key in self.cache_manager["override"]:
            intent_override = self.cache_manager["override"][override_key]
            intent_class = get_transformer(intent_override)
        else:
            intent_class = self._resolve_intent(X, y=y)

        return intent_class()
