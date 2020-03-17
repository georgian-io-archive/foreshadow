"""A utility class for the intents."""


class IntentType:
    """A utility class for the intents."""

    NUMERIC = "Numeric"
    CATEGORICAL = "Categorical"
    TEXT = "Text"
    DROPPABLE = "Droppable"

    _registered_types = [NUMERIC, CATEGORICAL, TEXT, DROPPABLE]

    @classmethod
    def is_valid(cls, intent):
        """Check if an intent is valid.

        Args:
            intent: user provided intent type

        Returns:
            bool: whether it's a valid intent

        """
        if intent in cls._registered_types:
            return True
        else:
            return False

    @classmethod
    def list_intents(cls):
        """List all the registered/valid intent types.

        Returns:
            a list of registered intents.

        """
        return cls._registered_types
