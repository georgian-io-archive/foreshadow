"""Cleaner module for handling the cleaning and shaping of data."""
from foreshadow.parallelprocessor import ParallelProcessor
from foreshadow.serializers import _make_deserializable, _make_serializable
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

    @classmethod
    def dict_deserialize(cls, data):
        params = _make_deserializable(data)
        import pdb

        pdb.set_trace()
        parallel_processor = cls.__reconstruct_parallel_process(params)
        reconstructed_params = {"_parallel_process": parallel_processor}
        ret_tf = cls()
        ret_tf.set_params(**reconstructed_params)
        return ret_tf

    @classmethod
    def __reconstruct_parallel_process(cls, data):
        import pdb

        pdb.set_trace()
        n_jobs = data["n_jobs"]
        transformer_weights = data["transformer_weights"]
        collapse_index = data["collapse_index"]

        transformer_list = []
        for i, transformation in enumerate(
            data["transformation_by_column_group"]
        ):
            group_name = "group: {}".format(str(i))
            dynamic_pipeline = list(transformation.values())[0]
            column_groups = list(transformation.keys())[0].split(",")
            transformer_list.append(
                (group_name, dynamic_pipeline, column_groups)
            )

        return ParallelProcessor(
            transformer_list, n_jobs, transformer_weights, collapse_index
        )

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
