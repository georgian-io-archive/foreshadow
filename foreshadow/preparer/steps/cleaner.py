"""Cleaner module for handling the cleaning and shaping of data."""
from foreshadow.preparer.preparerstep import PreparerStep
from foreshadow.smart import Cleaner, Flatten


class CleanerMapper(PreparerStep):
    """Determine and perform best data cleaning step."""

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
        import pdb

        pdb.set_trace()
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

    def __repr__(self):
        """Return string representation of this object with parent params.

        Returns:
            See above.

        """
        r = super().__repr__()
        preparer_params = self._preparer_params()
        preparer_params = {p: getattr(self, p, None) for p in preparer_params}
        preparer_print = ", ".join(
            ["{}={}".format(k, v) for k, v in preparer_params.items()]
        )
        return r[:-1] + preparer_print + ")"
