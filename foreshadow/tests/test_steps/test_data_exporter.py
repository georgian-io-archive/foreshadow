"""Test Data Exporter"""

from foreshadow.cachemanager import CacheManager
from foreshadow.steps import DataExporterMapper
from foreshadow.utils import ConfigKey


def test_data_exporter_fit_transform():
    export_path = "data_export.csv"
    cache_manager = CacheManager()
    cache_manager["config"][ConfigKey.PROCESSED_DATA_EXPORT_PATH] = export_path

    exporter = DataExporterMapper(cache_manager=cache_manager)

    from sklearn.datasets import load_breast_cancer
    import pandas as pd

    cancer = load_breast_cancer()
    cancerX_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

    processed_df = exporter.fit_transform(X=cancerX_df)

    pd.testing.assert_frame_equal(processed_df, cancerX_df)

    with open(export_path, "r") as fopen:
        exported_df = pd.read_csv(fopen)
        pd.testing.assert_frame_equal(processed_df, exported_df)
