"""Foreshadow specific exceptions."""


class InverseUnavailable(Exception):
    """Raised when an inverse transform is unavailable.

    An example of when this might occur is when empty data is passed into a
    transformer which thus, cannot invert said transformation.

    """

    pass


class TransformerNotFound(Exception):
    """Raised when a transformer cannot be found in the registry."""

    pass
