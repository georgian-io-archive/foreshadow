"""Intents."""

from foreshadow.concrete.internals.intents.base import BaseIntent
from foreshadow.concrete.internals.intents.categorical import Categoric
from foreshadow.concrete.internals.intents.numeric import Numeric
from foreshadow.concrete.internals.intents.text import Text


__all__ = ["Categoric", "Numeric", "Text", "BaseIntent"]
