"""PrepareStep that exports the processed data before sending to Estimator."""
from foreshadow.logging import logging
from foreshadow.utils import AcceptedKey, ConfigKey, DefaultConfig

from .preparerstep import PreparerStep


class DataExporterMapper(PreparerStep):
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

    def fit(self, X, *args, **kwargs):
        """Fit this step.

        calls underlying parallel process.

        Args:
            X: input DataFrame
            *args: args to _fit
            **kwargs: kwargs to _fit

        Returns:
            transformed data handled by Pipeline._fit

        """
        return self

    def fit_transform(self, X, *args, **kwargs):
        """Fit then transform a dataframe.

        Side-affect: export the dataframe to disk as a csv file.

        Args:
            X: input DataFrame
            *args: args to _fit
            **kwargs: kwargs to _fit

        Returns:
            Result from .transform(), pass through.

        """
        return self.fit(X, *args, **kwargs).transform(X, is_train=True)

    def transform(self, X, *args, is_train=False, **kwargs):
        """Transform a dataframe.

        Side-affect: export the dataframe to disk as a csv file.

        Args:
            X: input DataFrame
            *args: args to .transform()
            is_train: whether this is training or testing process
            **kwargs: kwargs to .transform()

        Returns:
            Result from .transform(), pass through.

        """
        self._export_data(X, is_train=is_train)
        return X

    def _export_data(self, X, is_train=True):
        data_path = self._determine_export_path(is_train)
        X.to_csv(data_path, index=False)
        logging.info("Exported processed data to {}".format(data_path))

    def _determine_export_path(self, is_train):
        key_to_check = (
            ConfigKey.PROCESSED_TRAINING_DATA_EXPORT_PATH
            if is_train
            else ConfigKey.PROCESSED_TEST_DATA_EXPORT_PATH
        )

        if key_to_check not in self.cache_manager[AcceptedKey.CONFIG]:
            data_path = (
                DefaultConfig.PROCESSED_TRAINING_DATA_EXPORT_PATH
                if is_train
                else DefaultConfig.PROCESSED_TEST_DATA_EXPORT_PATH
            )
        else:
            data_path = self.cache_manager[AcceptedKey.CONFIG][key_to_check]
        return data_path
