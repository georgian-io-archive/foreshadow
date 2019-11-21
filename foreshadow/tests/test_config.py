"""Test config resolution."""

import pytest


def test_get_config_does_not_exists(mocker):
    from foreshadow.config import load_config

    mocker.patch("os.path.exists", return_value=False)
    mocker.patch("os.path.isfile", return_value=False)

    assert load_config("test") == {}


@pytest.mark.parametrize(
    "data",
    [("test:\n  - hello".encode(), {"test": ["hello"]}), ("".encode(), {})],
)
def test_get_config_exists(data, mocker):
    from foreshadow.config import load_config

    read_data, test_data = data

    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("os.path.isfile", return_value=True)
    m = mocker.mock_open(read_data=read_data)
    mocker.patch("builtins.open", m, create=True)

    assert load_config("test") == test_data


@pytest.mark.skip("config changed temporarily")
def test_get_config_only_sys():
    import pickle

    from foreshadow.config import config
    from foreshadow.utils.testing import get_file_path

    resolved = config.get_config()

    test_data_path = get_file_path("configs", "configs_default.pkl")

    # (If you change default configs) or file structure, you will need to
    # verify the outputs are correct manually and regenerate the pickle
    # truth file.
    # with open(test_data_path, "wb") as fopen:
    #     pickle.dump(config[cfg_hash], fopen)

    with open(test_data_path, "rb") as fopen:
        test_data = pickle.load(fopen)

    assert resolved == test_data


@pytest.mark.parametrize(
    "data",
    [
        ({}, {}, {}, "configs_empty.json"),
        ({"Cleaner": ["T1", "T2"]}, {}, {}, "configs_override1.json"),
        (
            {"Cleaner": ["T1", "T2"]},
            {"Cleaner": ["T3"]},
            {},
            "configs_override2.json",
        ),
        (
            {"Cleaner": ["T1", "T2"]},
            {"Cleaner": ["T3"]},
            {"Cleaner": ["T4"]},
            "configs_override3.json",
        ),
        (
            {"Cleaner": ["T1", "T2"]},
            {},
            {"Cleaner": ["T4"]},
            "configs_override4.json",
        ),
    ],
)
def test_get_config_overrides(data, mocker):
    import json

    from foreshadow.config import config
    from foreshadow.utils.testing import get_file_path

    from functools import partial

    def mock_load_config(base, d1, d2):
        if base == "USER":
            return d1
        else:
            return d2

    framework, user, local, test_data_fname = data

    mock_load_config = partial(mock_load_config, d1=user, d2=local)

    mocker.patch(
        "foreshadow.config.config.system_config", return_value=framework
    )
    mocker.patch("foreshadow.config.get_config_path", return_value="USER")
    mocker.patch("os.path.abspath", return_value="LOCAL")
    mocker.patch("foreshadow.config.load_config", side_effect=mock_load_config)
    mocker.patch("foreshadow.config.get_transformer", side_effect=lambda x: x)

    # Clear the config cache
    config.clear()

    resolved = config.get_config()

    test_data_path = get_file_path("configs", test_data_fname)

    # # This shouldn't need to be done again (unless re-factor)
    # with open(test_data_path, 'w+') as fopen:
    #     json.dump(config[cfg_hash], fopen, indent=4)

    with open(test_data_path, "r") as fopen:
        test_data = json.load(fopen)

    assert resolved == test_data


# def test_cfg_caching(mocker):
#     pass  # TODO: write tests for this.
