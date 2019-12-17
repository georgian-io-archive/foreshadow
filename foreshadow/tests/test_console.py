import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, LinearSVR

from foreshadow.console import generate_model
from foreshadow.estimators import AutoEstimator
from foreshadow.utils import EstimatorFamily, ProblemType
from foreshadow.utils.testing import get_file_path


@pytest.mark.skip("level 2 is not implemented.")
def test_console_generate_level2():
    pass


def test_console_get_method_default_regression():
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

    result = get_method(
        None,
        y_train,
        family=EstimatorFamily.LINEAR,
        problem_type=ProblemType.REGRESSION,
    )

    assert isinstance(result, LinearRegression)


def test_console_get_method_default_classification():
    import pandas as pd

    from foreshadow.console import get_method

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    cancer = load_breast_cancer()
    X_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y_df = pd.DataFrame(cancer.target, columns=["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.2
    )

    result = get_method(
        None,
        y_train,
        family=EstimatorFamily.LINEAR,
        problem_type=ProblemType.CLASSIFICATION,
    )

    assert isinstance(result, LogisticRegression)


def test_console_get_method_override():
    # TODO may not need this method in the future if we decide to not
    #  allow the user to provide method override as it opens a lot of
    #  potential issues.
    from foreshadow.console import get_method

    from sklearn.linear_model import LogisticRegression

    result = get_method("LogisticRegression", None)

    assert isinstance(result, LogisticRegression)


def test_console_get_method_error():
    # TODO may not need this test. Same reason above.
    from foreshadow.console import get_method

    with pytest.raises(ValueError) as e:
        get_method("InvalidRegression", None)

    assert "Invalid method." in str(e.value)


@pytest.mark.parametrize(
    "filename, family, y_var, problem_type, estimator",
    [
        (
            "breast_cancer.csv",
            EstimatorFamily.LINEAR,
            "target",
            ProblemType.CLASSIFICATION,
            LogisticRegression,
        ),
        (
            "boston_housing.csv",
            EstimatorFamily.LINEAR,
            "medv",
            ProblemType.REGRESSION,
            LinearRegression,
        ),
        (
            "breast_cancer.csv",
            EstimatorFamily.SVM,
            "target",
            ProblemType.CLASSIFICATION,
            LinearSVC,
        ),
        (
            "boston_housing.csv",
            EstimatorFamily.SVM,
            "medv",
            ProblemType.REGRESSION,
            LinearSVR,
        ),
        (
            "breast_cancer.csv",
            EstimatorFamily.RF,
            "target",
            ProblemType.CLASSIFICATION,
            RandomForestClassifier,
        ),
        (
            "boston_housing.csv",
            EstimatorFamily.RF,
            "medv",
            ProblemType.REGRESSION,
            RandomForestRegressor,
        ),
        (
            "breast_cancer.csv",
            EstimatorFamily.NN,
            "target",
            ProblemType.CLASSIFICATION,
            MLPClassifier,
        ),
        (
            "boston_housing.csv",
            EstimatorFamily.NN,
            "medv",
            ProblemType.REGRESSION,
            MLPRegressor,
        ),
    ],
)
def test_console_generate_and_execute_model(
    filename, family, y_var, problem_type, estimator
):
    from foreshadow.console import generate_model, execute_model

    data_path = get_file_path("data", filename)

    args = ["--family", family, data_path, y_var, problem_type]

    model = generate_model(args)

    assert isinstance(model[0].estimator, estimator)

    execute_model(*model)


@pytest.mark.parametrize(
    "filename, y_var, problem_type, estimator",
    [
        ("boston_housing.csv", "medv", ProblemType.REGRESSION, AutoEstimator),
        (
            "breast_cancer.csv",
            "target",
            ProblemType.CLASSIFICATION,
            AutoEstimator,
        ),
    ],
)
def test_console_generate_level3(filename, y_var, problem_type, estimator):
    data_path = get_file_path("data", filename)

    args = ["--level", "3", data_path, y_var, problem_type]

    model = generate_model(args)

    assert isinstance(model[0].estimator, AutoEstimator)


def test_console_parse_args_multiprocess():
    from foreshadow.console import process_argument

    data_path = get_file_path("data", "boston_housing.csv")

    args = ["--level", "1", data_path, "medv", "regression"]
    cargs = process_argument(args)
    assert cargs.multiprocess is False

    args = ["--level", "1", "--multiprocess", data_path, "medv", "regression"]
    cargs = process_argument(args)
    assert cargs.multiprocess is True


def test_console_generate_invalid():
    from foreshadow.console import generate_model

    data_path = get_file_path("data", "boston_housing.csv")

    args = ["--level", "5", data_path, "medv", "regression"]

    with pytest.raises(ValueError) as e:
        generate_model(args)

    assert "Invalid Level" in str(e.value)


def test_console_generate_invalid_target():
    from foreshadow.console import generate_model

    data_path = get_file_path("data", "boston_housing.csv")

    args = ["--level", "5", data_path, "badtarget", "regression"]

    with pytest.raises(ValueError) as e:
        generate_model(args)

    assert "Invalid target variable" in str(e.value)


def test_console_generate_invalid_file():
    from foreshadow.console import generate_model

    data_path = get_file_path("data", "missing_file.csv")
    args = ["--level", "5", data_path, "badtarget", "regression"]

    with pytest.raises(ValueError) as e:
        generate_model(args)

    assert "Failed to load file." in str(e.value)


def test_console_generate_ignore_method():
    from foreshadow.console import generate_model

    data_path = get_file_path("data", "boston_housing.csv")

    args = [
        "--level",
        "3",
        data_path,
        "medv",
        "regression",
        "--method",
        "method",
    ]

    with pytest.warns(UserWarning, match="Method will be ignored"):
        generate_model(args)


@pytest.mark.skip("console broken until level 2 is implemented")
def test_console_generate_ignore_time():
    from foreshadow.console import generate_model

    data_path = get_file_path("data", "boston_housing.csv")

    args = [data_path, "medv", "--level", "2", "--time", "20"]

    with pytest.warns(UserWarning, match="Time parameter not applicable"):
        generate_model(args)
