"""PrepareStep that exports the processed data before sending to Estimator."""
from foreshadow.logging import logging
from foreshadow.smart import DataExporter
from foreshadow.utils import ConfigKey, DefaultConfig

from .autointentmap import AutoIntentMixin
from .preparerstep import PreparerStep


class DataExporterMapper(PreparerStep, AutoIntentMixin):
    """Define the single step for FeatureExporter.

    Args:
        **kwargs: kwargs to PreparerStep initializer.

    """

    def __init__(self, **kwargs):
        """Define the single step for FeatureExporter.

        Args:
            **kwargs: kwargs to PreparerStep initializer.

        """
        super().__init__(**kwargs)

    def get_mapping(self, X):  # noqa
        return self.separate_cols(
            transformers=[
                [DataExporter(cache_manager=self.cache_manager)] for c in X
            ],
            cols=X.columns,
        )

    def fit_transform(self, X, y=None, **fit_params):
        """Fit then transform this PreparerStep.

        Side-affect: export the dataframe to disk as a csv file.

        Args:
            X: input DataFrame
            y: input labels
            **fit_params: kwarg params to fit

        Returns:
            Result from .transform(), pass through.

        """
        Xt = super().fit_transform(X, y, **fit_params)
        if (
            ConfigKey.PROCESSED_DATA_EXPORT_PATH
            not in self.cache_manager["config"]
        ):
            data_path = DefaultConfig.PROCESSED_DATA_EXPORT_PATH
        else:
            data_path = self.cache_manager["config"][
                ConfigKey.PROCESSED_DATA_EXPORT_PATH
            ]
        Xt.to_csv(data_path, index=False)
        logging.info("Exported processed data to {}".format(data_path))
        return Xt
