"""Test Data Exporter"""

import pandas as pd
import pytest

from foreshadow.cachemanager import CacheManager
from foreshadow.steps import DataExporterMapper
from foreshadow.utils import AcceptedKey, ConfigKey, DefaultConfig


def _assert_common(export_path, processed_df, cancerX_df):
    pd.testing.assert_frame_equal(processed_df, cancerX_df)

    with open(export_path, "r") as fopen:
        exported_df = pd.read_csv(fopen)
        pd.testing.assert_frame_equal(processed_df, exported_df)


def _prepare_data_common():
    from sklearn.datasets import load_breast_cancer

    cancer = load_breast_cancer()
    return pd.DataFrame(cancer.data, columns=cancer.feature_names)


def test_data_exporter_fit_transform(tmpdir):
    export_path = tmpdir.join("data_export_training.csv")
    cache_manager = CacheManager()
    cache_manager[AcceptedKey.CONFIG][
        ConfigKey.PROCESSED_TRAINING_DATA_EXPORT_PATH
    ] = export_path

    exporter = DataExporterMapper(cache_manager=cache_manager)

    df = _prepare_data_common()
    processed_df = exporter.fit_transform(X=df)
    _assert_common(export_path, processed_df, df)


def test_data_exporter_transform(tmpdir):
    export_path = tmpdir.join("data_export_test.csv")
    cache_manager = CacheManager()
    cache_manager[AcceptedKey.CONFIG][
        ConfigKey.PROCESSED_TEST_DATA_EXPORT_PATH
    ] = export_path

    exporter = DataExporterMapper(cache_manager=cache_manager)

    df = _prepare_data_common()
    # Need to fit before transform, even though this step doesn't fit
    # anything. This is to stay consistent with all other transformers.
    _ = exporter.fit(X=df)
    processed_df = exporter.transform(X=df)
    _assert_common(export_path, processed_df, df)


@pytest.mark.parametrize("is_train", [True, False])
def test_data_exporter_determine_export_path_default(is_train):
    cache_manager = CacheManager()
    exporter = DataExporterMapper(cache_manager=cache_manager)

    data_path = exporter._determine_export_path(is_train=is_train)
    expected_data_path = (
        DefaultConfig.PROCESSED_TRAINING_DATA_EXPORT_PATH
        if is_train
        else DefaultConfig.PROCESSED_TEST_DATA_EXPORT_PATH
    )
    assert data_path == expected_data_path


@pytest.mark.parametrize(
    "is_train, user_specified_path",
    [
        (True, "processed_training_data.csv"),
        (False, "processed_test_data.csv"),
    ],
)
def test_data_exporter_determine_export_path_user_specified(
    is_train, user_specified_path
):
    cache_manager = CacheManager()
    key = (
        ConfigKey.PROCESSED_TRAINING_DATA_EXPORT_PATH
        if is_train
        else ConfigKey.PROCESSED_TEST_DATA_EXPORT_PATH
    )

    cache_manager[AcceptedKey.CONFIG][key] = user_specified_path

    exporter = DataExporterMapper(cache_manager=cache_manager)

    data_path = exporter._determine_export_path(is_train=is_train)
    expected_data_path = user_specified_path

    assert data_path == expected_data_path
