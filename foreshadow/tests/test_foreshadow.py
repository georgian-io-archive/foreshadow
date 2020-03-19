"""Test the Foreshadow class."""

import pytest
from sklearn.model_selection import train_test_split

from foreshadow.foreshadow import Foreshadow
from foreshadow.intents import IntentType
from foreshadow.utils import AcceptedKey, ProblemType
from foreshadow.utils.testing import get_file_path


def check_slow():
    import os

    return os.environ.get("FORESHADOW_TESTS") != "ALL"


slow = pytest.mark.skipif(
    check_slow(), reason="Skipping long-runnning integration tests"
)


def test_foreshadow_defaults():
    from foreshadow.foreshadow import Foreshadow
    from foreshadow.preparer import DataPreparer
    from foreshadow.estimators import AutoEstimator
    from foreshadow.estimators import EstimatorWrapper

    foreshadow = Foreshadow(problem_type=ProblemType.CLASSIFICATION)
    # defaults
    assert (
        isinstance(foreshadow.X_preparer, DataPreparer)
        and isinstance(foreshadow.y_preparer, DataPreparer)
        and isinstance(foreshadow.estimator_wrapper, EstimatorWrapper)
        and isinstance(foreshadow.estimator, AutoEstimator)
        # and foreshadow.optimizer is None
        and foreshadow.pipeline is None
        and foreshadow.data_columns is None
    )


def test_foreshadow_estimator_custom():
    from foreshadow.foreshadow import Foreshadow
    from foreshadow.base import BaseEstimator

    estimator = BaseEstimator()
    foreshadow = Foreshadow(
        problem_type=ProblemType.CLASSIFICATION, estimator=estimator
    )
    assert isinstance(foreshadow.estimator, BaseEstimator)


def test_foreshadow_estimator_error():
    from foreshadow.foreshadow import Foreshadow

    estimator = "Invalid"
    with pytest.raises(ValueError) as e:
        _ = Foreshadow(
            problem_type=ProblemType.CLASSIFICATION, estimator=estimator
        )

    assert str(e.value) == "Invalid value passed as estimator"


def test_foreshadow_y_preparer(mocker):
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from foreshadow.foreshadow import Foreshadow
    import pandas as pd

    np.random.seed(0)

    y_pipeline = Pipeline([("yohe", StandardScaler())])
    setattr(y_pipeline, "pipeline", y_pipeline)
    estimator = LinearRegression()

    X = pd.DataFrame(
        np.array([0] * 50 + [1] * 50).reshape((-1, 1)), columns=["col1"]
    )
    y = pd.DataFrame(
        np.random.normal(100, 10, 100).reshape((-1, 1)), columns=["y"]
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Let foreshadow set to defaults, we will overwrite them
    y_preparer = mocker.PropertyMock(return_value=y_pipeline)
    mocker.patch.object(Foreshadow, "y_preparer", y_preparer)
    foreshadow = Foreshadow(
        problem_type=ProblemType.REGRESSION, estimator=estimator
    )
    foreshadow.fit(X_train, y_train)
    foreshadow_predict = foreshadow.predict(X_test)
    foreshadow_score = foreshadow.score(X_test, y_test)
    expected_predict = np.array(
        [
            [102.19044770619593],
            [102.19044770619593],
            [102.19044770619593],
            [100.05275170774354],
            [102.19044770619593],
            [102.19044770619593],
            [102.19044770619593],
            [102.19044770619593],
            [100.05275170774354],
            [100.05275170774354],
        ]
    )
    expected_score = -0.3576910440975052

    assert np.allclose(foreshadow_predict, expected_predict)
    assert np.allclose(foreshadow_score, expected_score)


def test_foreshadow_predict_before_fit():
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from foreshadow.foreshadow import Foreshadow

    np.random.seed(0)
    estimator = LinearRegression()
    X = np.arange(200).reshape((-1, 2))
    y = np.random.normal(0, 1, 100).reshape((-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    foreshadow = Foreshadow(
        problem_type=ProblemType.REGRESSION, estimator=estimator
    )

    with pytest.raises(ValueError) as e:
        _ = foreshadow.predict(X_test)

    assert str(e.value) == "Foreshadow has not been fit yet"


@pytest.mark.skip("borken until parameter optimization is implemented")
def test_foreshadow_param_optimize_fit(mocker):
    import pandas as pd
    from foreshadow.base import BaseEstimator, TransformerMixin
    from sklearn.model_selection._search import BaseSearchCV

    from foreshadow.foreshadow import Foreshadow

    boston_path = get_file_path("data", "boston_housing.csv")
    data = pd.read_csv(boston_path)

    class DummyRegressor(BaseEstimator, TransformerMixin):
        def fit(self, X, y):
            return self

    class DummySearch(BaseSearchCV):
        def __init__(self, estimator, params):
            self.best_estimator_ = estimator

        def fit(self, X, y=None, **fit_params):
            return self

    class DummyDataPreparer(BaseEstimator, TransformerMixin):
        def fit(self, X, y):
            return self

    mocker.patch(
        "foreshadow.preparer.DataPreparer", return_value=DummyDataPreparer
    )

    fs = Foreshadow(
        problem_type=ProblemType.REGRESSION,
        estimator=DummyRegressor(),
        optimizer=DummySearch,
    )
    x = data.drop(["medv"], axis=1, inplace=False)
    y = data[["medv"]]

    fs.fit(x, y)
    assert isinstance(fs.pipeline.steps[-1][1].estimator, DummyRegressor)

    fs2 = Foreshadow(
        problem_type=ProblemType.REGRESSION,
        X_preparer=False,
        y_preparer=False,
        estimator=DummyRegressor(),
        optimizer=DummySearch,
    )

    fs2.fit(x, y)
    assert isinstance(fs2.pipeline.steps[-1][1], DummyRegressor)


@pytest.mark.skip("broken until parameter optimization is working")
def test_foreshadow_param_optimize():  # TODO: Make this test faster
    import pickle
    import json

    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV

    from foreshadow.foreshadow import Foreshadow
    from foreshadow.preparer import DataPreparer
    from foreshadow.optimizers.param_mapping import param_mapping

    boston_path = get_file_path("data", "boston_housing.csv")
    test_json_path = get_file_path("configs", "optimizer_test.json")

    truth_path = get_file_path("configs", "search_space_optimize.pkl")

    data = pd.read_csv(boston_path)
    js = json.load(open(test_json_path, "r"))

    fs = Foreshadow(
        DataPreparer(from_json=js),
        False,
        LinearRegression(),
        ProblemType.REGRESSION,
        GridSearchCV,
    )

    fs.pipeline = Pipeline(
        [("preparer", fs.X_preparer), ("estimator", fs.estimator)]
    )

    x = data.drop(["medv"], axis=1, inplace=False)
    y = data[["medv"]]

    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.25)

    results = param_mapping(fs.pipeline, x_train, y_train)

    # (If you change default configs) or file structure, you will need to
    # verify the outputs are correct manually and regenerate the pickle
    # truth file.
    truth = pickle.load(open(truth_path, "rb"))

    assert results[0].keys() == truth[0].keys()


@pytest.mark.skip("broken until parameter optimization is implemented.")
def test_foreshadow_param_optimize_no_config():
    import pickle

    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    from foreshadow.foreshadow import Foreshadow
    from foreshadow.preparer import DataPreparer
    from foreshadow.optimizers.param_mapping import param_mapping

    boston_path = get_file_path("data", "boston_housing.csv")
    test_path = get_file_path("configs", "search_space_no_cfg.pkl")

    data = pd.read_csv(boston_path)

    fs = Foreshadow(
        DataPreparer(),
        False,
        LinearRegression(),
        ProblemType.REGRESSION,
        GridSearchCV,
    )

    fs.pipeline = Pipeline(
        [("preparer", fs.X_preparer), ("estimator", fs.estimator)]
    )

    x = data.drop(["medv"], axis=1, inplace=False)
    y = data[["medv"]]

    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.25)

    results = param_mapping(fs.pipeline, x_train, y_train)

    truth = pickle.load(open(test_path, "rb"))

    assert results[0].keys() == truth[0].keys()


@pytest.mark.skip("broken until parameter optimization is implemented")
def test_foreshadow_param_optimize_no_combinations():
    import pickle

    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    from foreshadow.foreshadow import Foreshadow
    from foreshadow.preparer import DataPreparer
    from foreshadow.cachemanager import CacheManager

    boston_path = get_file_path("data", "boston_housing.csv")
    test_path = get_file_path("configs", "search_space_no_combo.pkl")

    data = pd.read_csv(boston_path)

    fs = Foreshadow(
        DataPreparer(cache_manager=CacheManager(), from_json={}),
        False,
        LinearRegression(),
        ProblemType.REGRESSION,
        GridSearchCV,
    )

    fs.pipeline = Pipeline(
        [("preprocessor", fs.X_preparer), ("estimator", fs.estimator)]
    )

    x = data.drop(["medv"], axis=1, inplace=False)
    y = data[["medv"]]

    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.25)

    results = param_mapping(fs.pipeline, x_train, y_train)  # noqa: F821

    truth = pickle.load(open(test_path, "rb"))

    assert results[0].keys() == truth[0].keys()


@pytest.mark.skip("broken until parameter optimization is implemented")
def test_foreshadow_param_optimize_invalid_array_idx():
    import json

    import pandas as pd

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    from foreshadow.foreshadow import Foreshadow
    from foreshadow.preparer import DataPreparer
    from foreshadow.cachemanager import CacheManager

    boston_path = get_file_path("data", "boston_housing.csv")
    test_path = get_file_path("configs", "invalid_optimizer_config.json")

    data = pd.read_csv(boston_path)
    cfg = json.load(open(test_path, "r"))

    fs = Foreshadow(
        DataPreparer(CacheManager(), from_json=cfg),
        False,
        LinearRegression(),
        ProblemType.REGRESSION,
        GridSearchCV,
    )

    fs.pipeline = Pipeline(
        [("preprocessor", fs.X_preparer), ("estimator", fs.estimator)]
    )

    x = data.drop(["medv"], axis=1, inplace=False)
    y = data[["medv"]]

    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.25)

    with pytest.raises(ValueError) as e:
        param_mapping(fs.pipeline, x_train, y_train)  # noqa: F821

    assert str(e.value).startswith("Attempted to index list")


@pytest.mark.skip("broken until parameter optimization is implemented")
def test_foreshadow_param_optimize_invalid_dict_key():
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    from foreshadow.foreshadow import Foreshadow
    from foreshadow.preparer import DataPreparer
    from foreshadow.cachemanager import CacheManager

    boston_path = get_file_path("data", "boston_housing.csv")

    data = pd.read_csv(boston_path)

    fs = Foreshadow(
        DataPreparer(
            cache_manager=CacheManager(),
            from_json={"combinations": [{"fake.fake": "[1,2]"}]},
        ),
        False,
        LinearRegression(),
        ProblemType.REGRESSION,
        GridSearchCV,
    )

    fs.pipeline = Pipeline(
        [("preprocessor", fs.X_preparer), ("estimator", fs.estimator)]
    )

    x = data.drop(["medv"], axis=1, inplace=False)
    y = data[["medv"]]

    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.25)

    with pytest.raises(ValueError) as e:
        param_mapping(fs.pipeline, x_train, y_train)  # noqa: F821

    assert str(e.value) == "Invalid JSON Key fake in {}"


def test_core_foreshadow_example_regression():
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_boston
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from foreshadow.foreshadow import Foreshadow

    np.random.seed(0)
    boston = load_boston()
    bostonX_df = pd.DataFrame(boston.data, columns=boston.feature_names)
    bostony_df = pd.DataFrame(boston.target, columns=["target"])
    X_train, X_test, y_train, y_test = train_test_split(
        bostonX_df, bostony_df, test_size=0.2
    )
    model = Foreshadow(
        estimator=LinearRegression(), problem_type=ProblemType.REGRESSION
    )
    model.fit(X_train, y_train)
    score = r2_score(y_test, model.predict(X_test))
    print("Boston score: %f" % score)


def test_core_foreshadow_example_classification():
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from foreshadow.foreshadow import Foreshadow

    np.random.seed(0)
    iris = load_iris()
    irisX_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    irisy_df = pd.DataFrame(iris.target, columns=["target"])
    X_train, X_test, y_train, y_test = train_test_split(
        irisX_df, irisy_df, test_size=0.2
    )

    model = Foreshadow(
        estimator=LogisticRegression(), problem_type=ProblemType.CLASSIFICATION
    )
    model.fit(X_train, y_train)

    score = f1_score(y_test, model.predict(X_test), average="weighted")
    print("Iris score: %f" % score)


@pytest.mark.parametrize("problem_type", [None, "Unknown"])
def test_foreshadow_unknown_problem_type(problem_type):
    from foreshadow.foreshadow import Foreshadow

    with pytest.raises(ValueError) as e:
        _ = Foreshadow(problem_type=problem_type)

    assert "Unknown Problem Type" in str(e.value)


@pytest.mark.parametrize("deep", [True, False])
def test_foreshadow_get_params_keys(deep):
    """Test that the desired keys show up for the Foreshadow object.

    Args:
        deep: deep param to get_params

    """
    from foreshadow.foreshadow import Foreshadow

    fs = Foreshadow(problem_type=ProblemType.CLASSIFICATION)
    params = fs.get_params(deep=deep)

    desired_keys = ["problem_type", "estimator", "data_columns"]
    for key in desired_keys:
        assert key in params


def test_foreshadow_serialization_breast_cancer_non_auto_estimator():
    from foreshadow.foreshadow import Foreshadow
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    np.random.seed(1337)

    cancer = load_breast_cancer()
    cancerX_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    cancery_df = pd.DataFrame(cancer.target, columns=["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        cancerX_df, cancery_df, test_size=0.2
    )

    shadow = Foreshadow(
        estimator=LogisticRegression(), problem_type=ProblemType.CLASSIFICATION
    )

    shadow.fit(X_train, y_train)
    score = shadow.score(X_test, y_test)
    print(score)


def test_foreshadow_serialization_adults_small_classification_override():
    from foreshadow.foreshadow import Foreshadow
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    np.random.seed(1337)

    data_path = get_file_path("data", "adult_small.csv")

    adult = pd.read_csv(data_path)
    X_df = adult.loc[:, "age":"workclass"]
    y_df = adult.loc[:, "class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.2
    )

    shadow = Foreshadow(
        estimator=LogisticRegression(), problem_type=ProblemType.CLASSIFICATION
    )
    shadow.fit(X_train, y_train)
    score1 = shadow.score(X_test, y_test)

    from foreshadow.intents import IntentType

    shadow.override_intent("age", IntentType.CATEGORICAL)
    shadow.override_intent("workclass", IntentType.CATEGORICAL)
    shadow.fit(X_train, y_train)

    assert shadow.get_intent("age") == IntentType.CATEGORICAL
    assert shadow.get_intent("workclass") == IntentType.CATEGORICAL
    score2 = shadow.score(X_test, y_test)
    print(score1, score2)


def test_foreshadow_adults_small_classification_override_upfront():
    from foreshadow.foreshadow import Foreshadow
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    np.random.seed(1337)

    data_path = get_file_path("data", "adult_small.csv")

    adult = pd.read_csv(data_path)
    X_df = adult.loc[:, "age":"workclass"]
    y_df = adult.loc[:, "class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.2
    )

    shadow = Foreshadow(
        estimator=LogisticRegression(), problem_type=ProblemType.CLASSIFICATION
    )

    from foreshadow.intents import IntentType

    shadow.override_intent("age", IntentType.CATEGORICAL)
    shadow.override_intent("workclass", IntentType.CATEGORICAL)
    shadow.fit(X_train, y_train)
    assert shadow.get_intent("age") == IntentType.CATEGORICAL
    assert shadow.get_intent("workclass") == IntentType.CATEGORICAL
    score = shadow.score(X_test, y_test)
    print(score)


def test_foreshadow_serialization_boston_housing_regression_multiprocessing(
    tmpdir
):
    from foreshadow.foreshadow import Foreshadow
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    np.random.seed(1337)

    boston = load_boston()
    X_df = pd.DataFrame(boston.data, columns=boston.feature_names)
    y_df = pd.DataFrame(boston.target, columns=["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.2
    )

    shadow = Foreshadow(
        estimator=LinearRegression(), problem_type=ProblemType.REGRESSION
    )

    shadow.configure_multiprocessing(n_job=-1)

    shadow.fit(X_train, y_train)

    score = shadow.score(X_test, y_test)
    print(score)


def train_test_split_local_file_common(file_path, X_start, X_end, target):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    np.random.seed(1337)

    adult = pd.read_csv(file_path)
    X_df = adult.loc[:, X_start:X_end]
    y_df = adult.loc[:, target]

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.2
    )

    return X_train, X_test, y_train, y_test


def construct_foreshadow_object_common(
    estimator=None, problem_type=None, estimator_kwargs={"max_time_mins": 1}
):
    if not estimator:
        from foreshadow.estimators import AutoEstimator

        estimator = AutoEstimator(
            problem_type=problem_type,
            auto="tpot",
            estimator_kwargs=estimator_kwargs,
        )
    shadow = Foreshadow(estimator=estimator, problem_type=problem_type)
    return shadow


def test_foreshadow_adults_classification():
    X_train, X_test, y_train, y_test = train_test_split_local_file_common(
        file_path=get_file_path("data", "adult.csv"),
        X_start="age",
        X_end="native-country",
        target="class",
    )
    shadow = construct_foreshadow_object_common(
        problem_type=ProblemType.CLASSIFICATION
    )

    shadow.fit(X_train, y_train)

    score = shadow.score(X_test, y_test)
    print(score)


def test_foreshadow_pickling_and_unpickling_unfitted(tmpdir):
    from foreshadow.foreshadow import Foreshadow
    from foreshadow.estimators import AutoEstimator

    estimator = AutoEstimator(
        problem_type=ProblemType.CLASSIFICATION,
        auto="tpot",
        estimator_kwargs={"max_time_mins": 1},
    )
    shadow = Foreshadow(
        estimator=estimator, problem_type=ProblemType.CLASSIFICATION
    )
    with pytest.raises(ValueError):
        shadow.pickle_fitted_pipeline(tmpdir.join("fitted_pipeline.p"))


def test_foreshadow_pickling_and_unpickling_non_tpot(tmpdir):
    from foreshadow.foreshadow import Foreshadow
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    np.random.seed(1337)

    cancer = load_breast_cancer()
    cancerX_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    cancery_df = pd.DataFrame(cancer.target, columns=["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        cancerX_df, cancery_df, test_size=0.2
    )

    from sklearn.linear_model import LogisticRegression

    shadow = Foreshadow(
        estimator=LogisticRegression(), problem_type=ProblemType.CLASSIFICATION
    )
    pickled_file_location = tmpdir.join("fitted_pipeline.p")
    shadow.fit(X_train, y_train)
    shadow.pickle_fitted_pipeline(pickled_file_location)

    import pickle

    with open(pickled_file_location, "rb") as fopen:
        pipeline = pickle.load(fopen)

    score1 = shadow.score(X_test, y_test)
    score2 = pipeline.score(X_test, y_test)

    import unittest

    assertions = unittest.TestCase("__init__")
    # given the randomness of the tpot algorithm and the short run
    # time we configured, there is no guarantee the performance can
    # converge. The test here aims to evaluate if both cases have
    # produced a reasonable score and the difference is small.
    assertions.assertAlmostEqual(score1, score2, places=2)


def test_foreshadow_pickling_and_unpickling_tpot(tmpdir):
    from foreshadow.foreshadow import Foreshadow
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    np.random.seed(1337)

    cancer = load_breast_cancer()
    cancerX_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    cancery_df = pd.DataFrame(cancer.target, columns=["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        cancerX_df, cancery_df, test_size=0.2
    )

    from foreshadow.estimators import AutoEstimator

    estimator = AutoEstimator(
        problem_type=ProblemType.CLASSIFICATION,
        auto="tpot",
        estimator_kwargs={"max_time_mins": 1},
    )

    shadow = Foreshadow(
        estimator=estimator, problem_type=ProblemType.CLASSIFICATION
    )
    pickled_file_location = tmpdir.join("fitted_pipeline.p")
    shadow.fit(X_train, y_train)
    shadow.pickle_fitted_pipeline(pickled_file_location)

    import pickle

    with open(pickled_file_location, "rb") as fopen:
        pipeline = pickle.load(fopen)

    score1 = shadow.score(X_test, y_test)
    score2 = pipeline.score(X_test, y_test)

    import unittest

    assertions = unittest.TestCase("__init__")
    # given the randomness of the tpot algorithm and the short run
    # time we configured, there is no guarantee the performance can
    # converge. The test here aims to evaluate if both cases have
    # produced a reasonable score and the difference is small.

    # Changing the decimal point to 1 due to failure on azure pipeline but
    # cannot be reproduced locally.
    assertions.assertAlmostEqual(score1, score2, places=2)


def test_foreshadow_configure_sampling():
    from foreshadow.foreshadow import Foreshadow
    from sklearn.linear_model import LogisticRegression
    from foreshadow.utils import ConfigKey

    shadow = Foreshadow(
        estimator=LogisticRegression(), problem_type=ProblemType.CLASSIFICATION
    )
    shadow.configure_sampling(enable_sampling=False)
    assert (
        shadow.X_preparer.cache_manager[AcceptedKey.CONFIG][
            ConfigKey.ENABLE_SAMPLING
        ]
        is False
    )

    shadow.configure_sampling(
        enable_sampling=True, sampling_fraction=0.3, replace=False
    )
    assert (
        shadow.X_preparer.cache_manager[AcceptedKey.CONFIG][
            ConfigKey.ENABLE_SAMPLING
        ]
        is True
    )
    assert (
        shadow.X_preparer.cache_manager[AcceptedKey.CONFIG][
            ConfigKey.SAMPLING_FRACTION
        ]
        == 0.3
    )
    assert (
        shadow.X_preparer.cache_manager[AcceptedKey.CONFIG][
            ConfigKey.SAMPLING_WITH_REPLACEMENT
        ]
        is False
    )


def test_foreshadow_sampling_performance_comparison():
    X_train, X_test, y_train, y_test = train_test_split_local_file_common(
        file_path=get_file_path("data", "adult_small.csv"),
        X_start="age",
        X_end="workclass",
        target="class",
    )
    shadow = construct_foreshadow_object_common(
        problem_type=ProblemType.CLASSIFICATION
    )
    import time

    start = time.time()
    shadow.X_preparer.fit_transform(X_train, y_train)
    end = time.time()
    time_taken1 = end - start

    shadow2 = construct_foreshadow_object_common(
        problem_type=ProblemType.CLASSIFICATION
    )
    shadow2.configure_sampling(enable_sampling=False)

    start = time.time()
    shadow2.X_preparer.fit_transform(X_train, y_train)
    end = time.time()
    time_taken2 = end - start

    # using sampling should be faster than without sampling on this dataset
    # as it has more than 40,000 rows.
    assert time_taken1 < time_taken2


@pytest.mark.parametrize(
    "filename,problem_type,X_start, X_end, target",
    [
        (
            "23380.csv",
            ProblemType.CLASSIFICATION,
            "INTERNODE_10",
            "INTERNODE_29",
            "target",
        )
    ],
)
def test_foreshadow_abort_on_empty_data_frame_after_cleaning(
    filename, problem_type, X_start, X_end, target
):
    from foreshadow.foreshadow import Foreshadow
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    np.random.seed(1337)

    data_path = get_file_path("data", filename)

    data = pd.read_csv(data_path)
    X_df = data.loc[:, X_start:X_end]
    y_df = data.loc[:, target]

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.2
    )

    from foreshadow.estimators import AutoEstimator

    estimator = AutoEstimator(
        problem_type=problem_type,
        auto="tpot",
        estimator_kwargs={"max_time_mins": 1},
    )

    shadow = Foreshadow(estimator=estimator, problem_type=problem_type)

    with pytest.raises(ValueError) as excinfo:
        shadow.fit(X_train, y_train)
    error_msg = (
        "All columns are dropped since they all have over 90% of "
        "missing values. Aborting foreshadow."
    )
    assert error_msg in str(excinfo.value)


@pytest.mark.parametrize(
    "filename,problem_type,X_start, X_end, target",
    [
        (
            "23380.csv",
            ProblemType.CLASSIFICATION,
            "TREE",
            "INTERNODE_29",
            "target",
        )
    ],
)
def test_foreshadow_integration_data_cleaner_can_drop(
    filename, problem_type, X_start, X_end, target, tmpdir
):
    from foreshadow.foreshadow import Foreshadow
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    np.random.seed(1337)

    data_path = get_file_path("data", filename)

    data = pd.read_csv(data_path)
    # local_file_folder = "examples"
    # data = pd.read_csv("/".join([local_file_folder, filename]))

    X_df = data.loc[:, X_start:X_end]
    y_df = data.loc[:, target]

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.2
    )

    from foreshadow.estimators import AutoEstimator

    estimator = AutoEstimator(
        problem_type=problem_type,
        auto="tpot",
        estimator_kwargs={"max_time_mins": 1},
    )

    shadow = Foreshadow(estimator=estimator, problem_type=problem_type)

    pickled_fitted_pipeline_location = tmpdir.join("fitted_pipeline.p")
    shadow.fit(X_train, y_train)
    shadow.pickle_fitted_pipeline(pickled_fitted_pipeline_location)

    import pickle

    with open(pickled_fitted_pipeline_location, "rb") as fopen:
        pipeline = pickle.load(fopen)

    # If there are new empty columns in the test set, the program should
    # not fail.
    X_test[X_start] = np.nan
    score1 = shadow.score(X_test, y_test)
    score2 = pipeline.score(X_test, y_test)

    import unittest

    assertions = unittest.TestCase("__init__")
    # given the randomness of the tpot algorithm and the short run
    # time we configured, there is no guarantee the performance can
    # converge. The test here aims to evaluate if both cases have
    # produced a reasonable score and the difference is small.
    # assert score1 > 0.76 and score2 > 0.76
    assertions.assertAlmostEqual(score1, score2, places=2)


@pytest.mark.parametrize(
    "filename,problem_type,X_start, X_end, target",
    [
        (
            "adult_small.csv",
            ProblemType.CLASSIFICATION,
            "age",
            "workclass",
            "class",
        )
    ],
)
def test_foreshadow_integration_adult_small_piclking_unpickling(
    filename, problem_type, X_start, X_end, target, tmpdir
):
    from foreshadow.foreshadow import Foreshadow
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    np.random.seed(1337)

    data_path = get_file_path("data", filename)

    data = pd.read_csv(data_path)
    X_df = data.loc[:, X_start:X_end]
    y_df = data.loc[:, target]

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.2
    )

    from foreshadow.estimators import AutoEstimator

    estimator = AutoEstimator(
        problem_type=problem_type,
        auto="tpot",
        estimator_kwargs={"max_time_mins": 1},
    )

    shadow = Foreshadow(estimator=estimator, problem_type=problem_type)

    pickled_fitted_pipeline_location = tmpdir.join("fitted_pipeline.p")
    shadow.fit(X_train, y_train)
    shadow.pickle_fitted_pipeline(pickled_fitted_pipeline_location)

    import pickle

    with open(pickled_fitted_pipeline_location, "rb") as fopen:
        pipeline = pickle.load(fopen)

    score1 = shadow.score(X_test, y_test)
    score2 = pipeline.score(X_test, y_test)

    import unittest

    assertions = unittest.TestCase("__init__")
    # given the randomness of the tpot algorithm and the short run
    # time we configured, there is no guarantee the performance can
    # converge. The test here aims to evaluate if both cases have
    # produced a reasonable score and the difference is small.
    # assert score1 > 0.76 and score2 > 0.76
    assertions.assertAlmostEqual(score1, score2, places=2)


def test_foreshadow_adults_small_user_provided_cleaner():
    from foreshadow.foreshadow import Foreshadow
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    np.random.seed(1337)

    data_path = get_file_path("data", "adult_small.csv")

    adult = pd.read_csv(data_path)
    X_df = adult.loc[:, "age":"workclass"]
    y_df = adult.loc[:, "class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.2
    )

    shadow = Foreshadow(
        estimator=LogisticRegression(), problem_type=ProblemType.CLASSIFICATION
    )

    from foreshadow.concrete.internals.cleaners.customizable_base import (
        CustomizableBaseCleaner,
    )

    def lowercase_row(row):
        """Lowercase a row.

        Args:
            row: string of text

        Returns:
            transformed row.

        """
        # Without using the customizable base cleaner, we have to explain
        # the meaning of the matched length. I don't know a good way to
        # explain it clearly without diving into the internal details yet.
        # return (row, 0) if row is None else (str(row).lower(), 1)

        return row if row is None else str(row).lower()

    class LowerCaseCleaner(CustomizableBaseCleaner):
        def __init__(self):
            super().__init__(transformation=lowercase_row)

        def metric_score(self, X: pd.DataFrame) -> float:
            """Calculate the matching metric score of the cleaner on this col.

            In this method, you specify the condition on when to apply the
            cleaner and calculate a confidence score between 0 and 1 where 1
            means 100% certainty to apply the transformation.

            Args:
                X: a column as a dataframe.

            Returns:
                the confidence score.

            """
            # The user needs to know what cleaners are provided so that
            # they don't create something duplicate or overlapping.
            column_name = list(X.columns)[0]
            if column_name == "workclass":
                return 1
            else:
                return 0

    shadow.register_customized_data_cleaner(data_cleaners=[LowerCaseCleaner])

    workclass_values = list(X_train["workclass"].unique())
    print(workclass_values)

    X_train_cleaned = shadow.X_preparer.steps[1][1].fit_transform(X_train)

    workclass_values_transformed = list(X_train_cleaned["workclass"].unique())
    for value in workclass_values_transformed:
        assert not any([c.isupper() for c in value])


def test_set_processed_data_export_path():
    from foreshadow.foreshadow import Foreshadow
    from sklearn.linear_model import LogisticRegression
    from foreshadow.utils import ConfigKey

    shadow = Foreshadow(
        estimator=LogisticRegression(), problem_type=ProblemType.CLASSIFICATION
    )
    processed_training_data_path = "datapath1.csv"
    shadow.set_processed_data_export_path(
        data_path=processed_training_data_path, is_train=True
    )
    assert (
        shadow.X_preparer.cache_manager[AcceptedKey.CONFIG][
            ConfigKey.PROCESSED_TRAINING_DATA_EXPORT_PATH
        ]
        == processed_training_data_path
    )

    processed_test_data_path = "datapath2.csv"
    shadow.set_processed_data_export_path(
        data_path=processed_test_data_path, is_train=False
    )
    assert (
        shadow.X_preparer.cache_manager[AcceptedKey.CONFIG][
            ConfigKey.PROCESSED_TEST_DATA_EXPORT_PATH
        ]
        == processed_test_data_path
    )


def test_foreshadow_titanic(tmpdir):
    import pandas as pd

    train_data = pd.read_csv(get_file_path("data", "titanic-train.csv"))
    X_train_df = train_data.loc[:, "Pclass":"Embarked"]
    y_train_df = train_data.loc[:, "Survived"]

    X_train_df = X_train_df.drop(columns=["SibSp", "Parch", "Cabin"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_train_df, y_train_df, test_size=0.2, random_state=42
    )

    from foreshadow.estimators import AutoEstimator

    estimator = AutoEstimator(
        problem_type=ProblemType.CLASSIFICATION,
        auto="tpot",
        estimator_kwargs={"max_time_mins": 1, "random_state": 42},
    )

    shadow = Foreshadow(
        estimator=estimator, problem_type=ProblemType.CLASSIFICATION
    )

    shadow.override_intent(column_name="Name", intent=IntentType.TEXT)
    shadow.fit(X_train, y_train)

    score = shadow.score(X_test, y_test)
    print(score)


def test_text_classification_foreshadow():
    import pandas as pd

    categories = [
        "alt.atheism",
        "soc.religion.christian",
        "comp.graphics",
        "sci.med",
    ]
    from sklearn.datasets import fetch_20newsgroups

    twenty_train = fetch_20newsgroups(
        subset="train", categories=categories, shuffle=True, random_state=42
    )
    X_train = pd.DataFrame(
        data=twenty_train.data,
        columns=["text"],
        index=list(range(len(twenty_train.data))),
    )
    y_train = pd.Series(
        data=twenty_train.target,
        name="category",
        index=list(range(len(twenty_train.target))),
    )

    twenty_test = fetch_20newsgroups(
        subset="test", categories=categories, shuffle=True, random_state=42
    )
    X_test = pd.DataFrame(data=twenty_test.data, columns=["text"])
    y_test = pd.Series(data=twenty_test.target, name="category")

    from foreshadow.estimators import AutoEstimator

    estimator = AutoEstimator(
        problem_type=ProblemType.CLASSIFICATION,
        auto="tpot",
        estimator_kwargs={"max_time_mins": 1, "random_state": 42},
    )

    shadow = Foreshadow(
        estimator=estimator, problem_type=ProblemType.CLASSIFICATION
    )

    shadow.fit(X_train, y_train)

    score = shadow.score(X_test, y_test)
    print(score)  # this gives about 87.5%


# def test_text_classification():
#     # This is a test on text classification for pure Sklearn pipeline so we
#     # can compare the result with foreshadow.
#     # Using the NB estimator the performance is around 83.5%.
#     categories = ['alt.atheism', 'soc.religion.christian',
#                   'comp.graphics', 'sci.med']
#     from sklearn.datasets import fetch_20newsgroups
#     twenty_train = fetch_20newsgroups(subset='train',
#                                       categories=categories, shuffle=True,
#                                       random_state=42)
#     print(twenty_train.target_names)
#
#     from sklearn.feature_extraction.text import CountVectorizer
#     count_vect = CountVectorizer()
#     X_train_counts = count_vect.fit_transform(twenty_train.data)
#     print(X_train_counts.shape)
#
#     from sklearn.feature_extraction.text import TfidfTransformer
#     tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
#     X_train_tf = tf_transformer.transform(X_train_counts)
#     print(X_train_tf.shape)
#
#     tfidf_transformer = TfidfTransformer()
#     X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#     print(X_train_tfidf.shape)
#
#     from sklearn.naive_bayes import MultinomialNB
#
#     from sklearn.pipeline import Pipeline
#     text_clf = Pipeline([
#         ('vect', CountVectorizer()),
#         ('tfidf', TfidfTransformer()),
#         ('clf', MultinomialNB()),
#     ])
#
#     text_clf.fit(twenty_train.data, twenty_train.target)
#
#     import numpy as np
#     twenty_test = fetch_20newsgroups(subset='test',
#                                      categories=categories, shuffle=True,
#                                      random_state=42)
#     docs_test = twenty_test.data
#     predicted = text_clf.predict(docs_test)
#     print(np.mean(predicted == twenty_test.target))
