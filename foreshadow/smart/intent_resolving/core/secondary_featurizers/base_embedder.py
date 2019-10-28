"""Class definition of the BaseEmbedder ABC."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseEmbedder(ABC):
    """
    Abstract base class to wrap text embedders from various sources.

    This wrapper standardizes the API for each text embedder by exposing
    three methods mentioned below.

    Attributes:
        embedder -- Object that converts text to embeddings.

    Abstract methods:
        load -- Load the embedder
        fit -- Fit the embedder to the training data
        transform -- Transfrom text to embeddings
    """

    def __init__(self):
        """Init function."""
        self.embedder = None

    @abstractmethod
    def load(self):
        """Load the embedder."""
        raise NotImplementedError

    @abstractmethod
    def fit(self):
        """Fit the emdedder."""
        raise NotImplementedError

    @abstractmethod
    def transform(self) -> pd.DataFrame:
        """Transform input text values into embeddings."""
        raise NotImplementedError

    @abstractmethod
    def serialize(self):
        """Return a serializable representation of an embedder."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def deserialize(serialization: dict):
        """Instantiate an object from a serialization."""
        raise NotImplementedError
