"""Cleaner module for handling the cleaning and shaping of data."""
from foreshadow.smart import Cleaner, Flatten

from .preparerstep import PreparerStep


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
        return self.separate_cols(
            transformers=[
                [
                    Flatten(cache_manager=self.cache_manager),
                    Cleaner(cache_manager=self.cache_manager),
                ]
                for c in X
            ],
            cols=X.columns,
        )

    def fit_transform(self, X, y=None, **fit_params):
        """Fit then transform this PreparerStep.

        Call the method in the super class and then drop all the empty columns.

        Args:
            X: input DataFrame
            y: input labels
            **fit_params: kwarg params to fit

        Returns:
            Result from .transform()

        """
        Xt = super().fit_transform(X=X, y=y, **fit_params)
        return Xt.dropna(how="all", axis=1)
