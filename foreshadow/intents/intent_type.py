"""A utility class for the intents."""

from ..intents import Categorical, Numeric, Text


class IntentType:
    """A utility class for the intents."""

    Numeric = Numeric.__name__
    Categorical = Categorical.__name__
    Text = Text.__name__

    _registered_types = [Numeric, Categorical, Text]

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
