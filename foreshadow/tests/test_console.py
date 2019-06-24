import pytest


def test_console_generate_ignore_method():
    from foreshadow.console import generate_model

    args = [
        "./foreshadow/tests/test_data/boston_housing.csv",
        "medv",
        "--level",
        "3",
        "--method",
        "method",
    ]

    with pytest.warns(UserWarning, match="Method will be ignored"):
        generate_model(args)


def test_console_generate_ignore_time():
    from foreshadow.console import generate_model

    args = [
        "./foreshadow/tests/test_data/boston_housing.csv",
        "medv",
        "--level",
        "2",
        "--time",
        "20",
    ]

    with pytest.warns(UserWarning, match="Time parameter not applicable"):
        generate_model(args)


def test_console_generate_invalid_file():
    from foreshadow.console import generate_model

    args = ["badfile.csv", "test"]

    with pytest.raises(ValueError) as e:
        generate_model(args)

    assert "Failed to load file." in str(e.value)


def test_console_generate_invalid_target():
    from foreshadow.console import generate_model

    args = ["./foreshadow/tests/test_data/boston_housing.csv", "badtarget"]

    with pytest.raises(ValueError) as e:
        generate_model(args)

    assert "Invalid target variable" in str(e.value)


def test_console_generate_default():
    from foreshadow.console import generate_model
    from sklearn.linear_model import LinearRegression

    args = ["./foreshadow/tests/test_data/boston_housing.csv", "medv"]

    model = generate_model(args)

    assert isinstance(model[0].estimator, LinearRegression)


def test_console_generate_invalid():
    from foreshadow.console import generate_model

    args = [
        "./foreshadow/tests/test_data/boston_housing.csv",
        "medv",
        "--level",
        "5",
    ]

    with pytest.raises(ValueError) as e:
        generate_model(args)

    assert "Invalid Level" in str(e.value)


def test_console_generate_level2():
    from foreshadow.console import generate_model
    from sklearn.linear_model import LinearRegression

    args = [
        "./foreshadow/tests/test_data/boston_housing.csv",
        "medv",
        "--level",
        "2",
    ]

    model = generate_model(args)

    assert isinstance(model[0].estimator, LinearRegression)


def test_console_generate_config():
    import json

    from foreshadow.console import generate_model

    args = [
        "./foreshadow/tests/test_data/boston_housing.csv",
        "medv",
        "--level",
        "2",
        "--x_config",
        "./foreshadow/tests/test_configs/override_multi_pipeline.json",
        "--y_config",
        "./foreshadow/tests/test_configs/override_multi_pipeline.json",
    ]

    model = generate_model(args)

    assert model[0].X_preprocessor.from_json == json.load(
        open(
            "./foreshadow/tests/test_configs/override_multi_pipeline.json", "r"
        )
    )
    assert model[0].y_preprocessor.from_json == json.load(
        open(
            "./foreshadow/tests/test_configs/override_multi_pipeline.json", "r"
        )
    )


def test_console_invalid_x_config():
    from foreshadow.console import generate_model

    args = [
        "./foreshadow/tests/test_data/boston_housing.csv",
        "medv",
        "--level",
        "2",
        "--x_config",
        "./foreshadow/tests/test_configs/invalid.json",
        "--y_config",
        "./foreshadow/tests/test_configs/override_multi_pipeline.json",
    ]

    with pytest.raises(ValueError) as e:
        generate_model(args)

    assert "Could not read X config file" in str(e.value)


def test_console_invalid_y_config():
    from foreshadow.console import generate_model

    args = [
        "./foreshadow/tests/test_data/boston_housing.csv",
        "medv",
        "--level",
        "2",
        "--x_config",
        "./foreshadow/tests/test_configs/override_multi_pipeline.json",
        "--y_config",
        "./foreshadow/tests/test_configs/invalid.json",
    ]

    with pytest.raises(ValueError) as e:
        generate_model(args)

    assert "Could not read y config file" in str(e.value)


def test_console_generate_level3():
    from foreshadow.estimators import AutoEstimator
    from foreshadow.console import generate_model

    args = [
        "./foreshadow/tests/test_data/boston_housing.csv",
        "medv",
        "--level",
        "3",
    ]

    model = generate_model(args)

    assert isinstance(model[0].estimator, AutoEstimator)


def test_console_execute():
    import pandas as pd

    from foreshadow.console import execute_model
    from foreshadow import Foreshadow

    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    boston = load_boston()
    X_df = pd.DataFrame(boston.data, columns=boston.feature_names)
    y_df = pd.DataFrame(boston.target, columns=["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.2
    )
    fs = Foreshadow(estimator=LinearRegression())

    results = execute_model(fs, X_train, y_train, X_test, y_test)

    assert set(["X_Model", "X_Summary", "y_Model", "y_summary"]) == set(
        results.keys()
    )


def test_console_get_method_default():
    import pandas as pd

    from foreshadow.console import get_method

    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    boston = load_boston()
    X_df = pd.DataFrame(boston.data, columns=boston.feature_names)
    y_df = pd.DataFrame(boston.target, columns=["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.2
    )

    result = get_method(None, X_train)

    assert isinstance(result, LinearRegression)


def test_console_get_method_override():
    from foreshadow.console import get_method

    from sklearn.linear_model import LogisticRegression

    result = get_method("LogisticRegression", None)

    assert isinstance(result, LogisticRegression)


def test_console_get_method_error():
    from foreshadow.console import get_method

    with pytest.raises(ValueError) as e:
        get_method("InvalidRegression", None)

    assert "Invalid method." in str(e.value)
