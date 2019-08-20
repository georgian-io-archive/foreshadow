"""Cleaner module for handling the cleaning and shaping of data."""
from foreshadow.serializers import _make_serializable
from foreshadow.smart import Cleaner, Flatten

from .preparerstep import PreparerStep


# flake8: noqa


class CleanerMapper(PreparerStep):
    """Determine and perform best data cleaning step."""

    def dict_serialize(self, deep=True):
        """Flake8 you are annoying during development..."""
        selected_params = self.get_params(deep=deep)["_parallel_process"]
        serialized = _make_serializable(
            selected_params, serialize_args=self.serialize_params
        )
        transformer_list = serialized["transformer_list"]
        serialized.pop("transformer_list")
        serialized["transformation_by_column_group"] = transformer_list
        return serialized

    def __init__(self, **kwargs):
        """Define the single step for CleanerMapper, using SmartCleaner.

        Args:
            **kwargs: kwargs to PreparerStep constructor.

        """
        super().__init__(**kwargs)

    def get_mapping(self, X):
        """Return the mapping of transformations for the CleanerMapper step.

        Args:
            X: input DataFrame.

        Returns:
            Mapping in accordance with super.

        """
        return self.separate_cols(
            transformers=[
                [
                    Flatten(column_sharer=self.column_sharer),
                    Cleaner(column_sharer=self.column_sharer),
                ]
                for c in X
            ],
            cols=X.columns,
        )
