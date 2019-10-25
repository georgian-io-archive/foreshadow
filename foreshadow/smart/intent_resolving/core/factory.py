"""GenericFactory class definition."""


class GenericFactory:
    """A generic factory to be used by other group of builder classes."""

    def __init__(self):
        """Init function."""
        self._builders = {}

    def register_builders(self, key: str, builder):
        """Register builder classes to factory instance."""
        self._builders[key] = builder

    def create(self, key: str, **kwargs):
        """
        Create a builder based on key provided.

        Arguments:
            key {str} -- Key to identify which registered builder to use.

        Raises:
            ValueError -- If builder `key` is not registered.

        Returns:
            A registered builder.
        """
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)

        return builder(**kwargs)
