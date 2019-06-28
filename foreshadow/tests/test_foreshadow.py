from unittest.mock import PropertyMock, patch

import pytest


def test_foreshadow_defaults():
    from foreshadow import Foreshadow
    from foreshadow import Preprocessor
    from foreshadow.estimators import AutoEstimator

    foreshadow = Foreshadow()
    # defaults
    assert (
        isinstance(foreshadow.X_preprocessor, Preprocessor)
        and isinstance(foreshadow.y_preprocessor, Preprocessor)
        and isinstance(foreshadow.estimator, AutoEstimator)
        and foreshadow.optimizer is None
        and foreshadow.pipeline is None
        and foreshadow.data_columns is None
    )


def test_foreshadow_X_preprocessor_false():
    from foreshadow import Foreshadow

    foreshadow = Foreshadow(X_preprocessor=False)
    assert foreshadow.X_preprocessor is None


def test_foreshadow_X_preprocessor_custom():
    from foreshadow import Foreshadow
    from foreshadow import Preprocessor

    preprocessor = Preprocessor()
    foreshadow = Foreshadow(X_preprocessor=preprocessor)
    assert type(foreshadow.X_preprocessor) == Preprocessor


def test_foreshadow_X_preprocessor_error():
    from foreshadow import Foreshadow

    preprocessor = "Invalid"
    with pytest.raises(ValueError) as e:
        _ = Foreshadow(X_preprocessor=preprocessor)

    assert str(e.value) == "Invalid value passed as X_preprocessor"


def test_foreshadow_y_preprocessor_false():
    from foreshadow import Foreshadow

    foreshadow = Foreshadow(y_preprocessor=False)
    assert foreshadow.y_preprocessor is None


def test_foreshadow_y_preprocessor_custom():
    from foreshadow import Foreshadow
    from foreshadow import Preprocessor

    preprocessor = Preprocessor()
    foreshadow = Foreshadow(y_preprocessor=preprocessor)
    assert type(foreshadow.y_preprocessor) == Preprocessor


def test_foreshadow_y_preprocessor_error():
    from foreshadow import Foreshadow

    preprocessor = "Invalid"
    with pytest.raises(ValueError) as e:
        _ = Foreshadow(y_preprocessor=preprocessor)

    assert str(e.value) == "Invalid value passed as y_preprocessor"


def test_foreshadow_estimator_custom():
    from foreshadow import Foreshadow
    from sklearn.base import BaseEstimator

    estimator = BaseEstimator()
    foreshadow = Foreshadow(estimator=estimator)
    assert isinstance(foreshadow.estimator, BaseEstimator)


def test_foreshadow_estimator_error():
    from foreshadow import Foreshadow

    estimator = "Invalid"
    with pytest.raises(ValueError) as e:
        _ = Foreshadow(estimator=estimator)

    assert str(e.value) == "Invalid value passed as estimator"


def test_foreshadow_optimizer_custom():
    from foreshadow import Foreshadow
    from sklearn.model_selection._search import BaseSearchCV
    from sklearn.base import BaseEstimator

    class DummySearch(BaseSearchCV):
        pass

    # Need custom estimator to avoid warning
    estimator = BaseEstimator()
    foreshadow = Foreshadow(estimator=estimator, optimizer=DummySearch)
    assert issubclass(foreshadow.optimizer, BaseSearchCV)


def test_foreshadow_optimizer_error_invalid():
    from foreshadow import Foreshadow

    optimizer = "Invalid"
    with pytest.raises(ValueError) as e:
        _ = Foreshadow(optimizer=optimizer)

    assert str(e.value) == "Invalid value passed as optimizer"


def test_foreshadow_optimizer_error_wrongclass():
    from foreshadow import Foreshadow

    optimizer = Foreshadow
    with pytest.raises(ValueError) as e:
        _ = Foreshadow(optimizer=optimizer)

    assert str(e.value) == "Invalid value passed as optimizer"


def test_foreshadow_warns_on_set_estimator_optimizer():
    from foreshadow import Foreshadow
    from sklearn.model_selection._search import BaseSearchCV

    class DummySearch(BaseSearchCV):
        pass

    with pytest.warns(Warning) as w:
        _ = Foreshadow(optimizer=DummySearch)

    assert str(w[0].message) == (
        "An automatic estimator cannot be used with an"
        " optimizer. Proceeding without use of optimizer"
    )


# patch to override type verification
@patch(
    "foreshadow.foreshadow.Foreshadow.X_preprocessor",
    create=True,
    new_callable=PropertyMock,
)
def test_foreshadow_custom_fit_estimate(X_preprocessor):
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from foreshadow import Foreshadow

    np.random.seed(0)

    X_pipeline = Pipeline([("xohe", OneHotEncoder())])
    setattr(X_pipeline, "pipeline", X_pipeline)
    estimator = LogisticRegression()

    X = np.array([0] * 50 + [1] * 50).reshape((-1, 1))
    y = np.array([0] * 50 + [1] * 50)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Let foreshadow set to defaults, we will overwrite them
    foreshadow = Foreshadow(y_preprocessor=False, estimator=estimator)
    X_preprocessor.return_value = X_pipeline
    foreshadow.fit(X_train, y_train)
    foreshadow_predict = foreshadow.predict(X_test)
    foreshadow_predict_proba = foreshadow.predict_proba(X_test)
    foreshadow_score = foreshadow.score(X_test, y_test)
    expected_predict = np.array([0, 1, 0, 1, 1, 1, 0, 1, 1, 1])
    expected_predict_proba = np.array(
        [
            [0.9414791454949417, 0.05852085450505827],
            [0.06331066362121573, 0.9366893363787843],
            [0.9414791454949417, 0.05852085450505827],
            [0.06331066362121573, 0.9366893363787843],
            [0.06331066362121573, 0.9366893363787843],
            [0.06331066362121573, 0.9366893363787843],
            [0.9414791454949417, 0.05852085450505827],
            [0.06331066362121573, 0.9366893363787843],
            [0.06331066362121573, 0.9366893363787843],
            [0.06331066362121573, 0.9366893363787843],
        ]
    )
    expected_score = 1.0

    assert np.allclose(foreshadow_predict, expected_predict)
    assert np.allclose(foreshadow_predict_proba, expected_predict_proba)
    assert np.allclose(foreshadow_score, expected_score)


@patch(
    "foreshadow.foreshadow.Foreshadow.y_preprocessor",
    create=True,
    new_callable=PropertyMock,
)
def test_foreshadow_y_preprocessor(y_preprocessor):
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from foreshadow import Foreshadow

    np.random.seed(0)

    y_pipeline = Pipeline([("yohe", StandardScaler())])
    setattr(y_pipeline, "pipeline", y_pipeline)
    estimator = LinearRegression()

    X = np.array([0] * 50 + [1] * 50).reshape((-1, 1))
    y = np.random.normal(100, 10, 100).reshape((-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Let foreshadow set to defaults, we will overwrite them
    foreshadow = Foreshadow(X_preprocessor=False, estimator=estimator)
    y_preprocessor.return_value = y_pipeline
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


def test_foreshadow_without_x_processor():
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from foreshadow import Foreshadow

    np.random.seed(0)
    estimator = LinearRegression()
    X = np.arange(200).reshape((-1, 1))
    y = np.random.normal(0, 1, 200).reshape((-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    foreshadow = Foreshadow(
        X_preprocessor=False, y_preprocessor=False, estimator=estimator
    )
    foreshadow.fit(X_train, y_train)
    foreshadow_predict = foreshadow.predict(X_test)
    expected_predict = np.array(
        [
            [0.21789584803659176],
            [-0.11658780412675052],
            [-0.008315639264628194],
            [-0.04698426957252905],
            [0.229496437128962],
            [0.09608966256670409],
            [-0.07791917381884972],
            [0.21402898500580167],
            [0.20629525894422152],
            [-0.023783091387788502],
            [0.06902162135117351],
            [0.10769025165907437],
            [0.0825556419589388],
            [0.16569319712092562],
            [-0.11852123564214556],
            [-0.11078750958056544],
            [0.028419559527877614],
            [-0.01604936532620832],
            [0.03035299104327266],
            [0.09222279953591403],
        ]
    )

    assert np.allclose(foreshadow_predict, expected_predict)


def test_foreshadow_predict_before_fit():
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from foreshadow import Foreshadow

    np.random.seed(0)
    estimator = LinearRegression()
    X = np.arange(200).reshape((-1, 2))
    y = np.random.normal(0, 1, 100).reshape((-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    foreshadow = Foreshadow(
        X_preprocessor=False, y_preprocessor=False, estimator=estimator
    )

    with pytest.raises(ValueError) as e:
        _ = foreshadow.predict(X_test)

    assert str(e.value) == "Foreshadow has not been fit yet"


def test_foreshadow_predict_diff_cols():
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from foreshadow import Foreshadow

    np.random.seed(0)
    estimator = LinearRegression()
    X = np.arange(200).reshape((-1, 2))
    y = np.random.normal(0, 1, 100).reshape((-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    foreshadow = Foreshadow(
        X_preprocessor=False, y_preprocessor=False, estimator=estimator
    )
    foreshadow.fit(X_train, y_train)

    with pytest.raises(ValueError) as e:
        _ = foreshadow.predict(X_test[:, :-1])

    assert (
        str(e.value) == "Predict must have the same columns as train columns"
    )


@patch("foreshadow.preprocessor.Preprocessor")
def test_foreshadow_param_optimize_fit(mock_p):
    import os

    import pandas as pd
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.model_selection._search import BaseSearchCV

    from foreshadow import Foreshadow

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )
    data = pd.read_csv(boston_path)

    class DummyRegressor(BaseEstimator, TransformerMixin):
        def fit(self, X, y):
            return self

    class DummySearch(BaseSearchCV):
        def __init__(self, estimator, params):
            self.best_estimator_ = estimator

        def fit(self, X, y=None, **fit_params):
            return self

    class DummyPreprocessor(BaseEstimator, TransformerMixin):
        def fit(self, X, y):
            return self

    mock_p.return_value = DummyPreprocessor()

    fs = Foreshadow(estimator=DummyRegressor(), optimizer=DummySearch)
    x = data.drop(["medv"], axis=1, inplace=False)
    y = data[["medv"]]

    fs.fit(x, y)
    assert isinstance(fs.pipeline.steps[-1][1].estimator, DummyRegressor)

    fs2 = Foreshadow(
        X_preprocessor=False,
        y_preprocessor=False,
        estimator=DummyRegressor(),
        optimizer=DummySearch,
    )

    fs2.fit(x, y)
    assert isinstance(fs2.pipeline.steps[-1][1], DummyRegressor)


def test_foreshadow_param_optimize():  # TODO: Make this test faster
    import os
    import pickle
    import json

    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV

    from foreshadow import Foreshadow
    from foreshadow.preprocessor import Preprocessor
    from foreshadow.optimizers.param_mapping import _param_mapping

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )
    test_json_path = os.path.join(
        os.path.dirname(__file__), "test_configs", "optimizer_test.json"
    )
    truth_path = os.path.join(
        os.path.dirname(__file__), "test_configs", "search_space_optimize.pkl"
    )

    data = pd.read_csv(boston_path)
    js = json.load(open(test_json_path, "r"))

    fs = Foreshadow(
        Preprocessor(from_json=js), False, LinearRegression(), GridSearchCV
    )

    fs.pipeline = Pipeline(
        [("preprocessor", fs.X_preprocessor), ("estimator", fs.estimator)]
    )

    x = data.drop(["medv"], axis=1, inplace=False)
    y = data[["medv"]]

    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.25)

    results = _param_mapping(fs.pipeline, x_train, y_train)
    truth = pickle.load(open(truth_path, "rb"))

    assert results[0].keys() == truth[0].keys()


def test_foreshadow_param_optimize_no_config():
    import os
    import pickle

    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    from foreshadow import Foreshadow
    from foreshadow.preprocessor import Preprocessor
    from foreshadow.optimizers.param_mapping import _param_mapping

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )
    test_path = os.path.join(
        os.path.dirname(__file__), "test_configs", "search_space_no_cfg.pkl"
    )

    data = pd.read_csv(boston_path)

    fs = Foreshadow(Preprocessor(), False, LinearRegression(), GridSearchCV)

    fs.pipeline = Pipeline(
        [("preprocessor", fs.X_preprocessor), ("estimator", fs.estimator)]
    )

    x = data.drop(["medv"], axis=1, inplace=False)
    y = data[["medv"]]

    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.25)

    results = _param_mapping(fs.pipeline, x_train, y_train)

    truth = pickle.load(open(test_path, "rb"))

    assert results[0].keys() == truth[0].keys()


def test_foreshadow_param_optimize_no_combinations():
    import os
    import pickle

    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    from foreshadow import Foreshadow
    from foreshadow.preprocessor import Preprocessor
    from foreshadow.optimizers.param_mapping import _param_mapping

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )
    test_path = os.path.join(
        os.path.dirname(__file__), "test_configs", "search_space_no_combo.pkl"
    )

    data = pd.read_csv(boston_path)

    fs = Foreshadow(
        Preprocessor(from_json={}), False, LinearRegression(), GridSearchCV
    )

    fs.pipeline = Pipeline(
        [("preprocessor", fs.X_preprocessor), ("estimator", fs.estimator)]
    )

    x = data.drop(["medv"], axis=1, inplace=False)
    y = data[["medv"]]

    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.25)

    results = _param_mapping(fs.pipeline, x_train, y_train)
    truth = pickle.load(open(test_path, "rb"))

    assert results[0].keys() == truth[0].keys()


def test_foreshadow_param_optimize_invalid_array_idx():
    import json
    import os

    import pandas as pd

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    from foreshadow import Foreshadow
    from foreshadow.preprocessor import Preprocessor
    from foreshadow.optimizers.param_mapping import _param_mapping

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )
    test_path = os.path.join(
        os.path.dirname(__file__),
        "test_configs",
        "invalid_optimizer_config.json",
    )

    data = pd.read_csv(boston_path)
    cfg = json.load(open(test_path, "r"))

    fs = Foreshadow(
        Preprocessor(from_json=cfg), False, LinearRegression(), GridSearchCV
    )

    fs.pipeline = Pipeline(
        [("preprocessor", fs.X_preprocessor), ("estimator", fs.estimator)]
    )

    x = data.drop(["medv"], axis=1, inplace=False)
    y = data[["medv"]]

    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.25)

    with pytest.raises(ValueError) as e:
        _param_mapping(fs.pipeline, x_train, y_train)

    assert str(e.value).startswith("Attempted to index list")


def test_foreshadow_param_optimize_invalid_dict_key():
    import os

    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    from foreshadow import Foreshadow
    from foreshadow.preprocessor import Preprocessor
    from foreshadow.optimizers.param_mapping import _param_mapping

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )

    data = pd.read_csv(boston_path)

    fs = Foreshadow(
        Preprocessor(from_json={"combinations": [{"fake.fake": "[1,2]"}]}),
        False,
        LinearRegression(),
        GridSearchCV,
    )

    fs.pipeline = Pipeline(
        [("preprocessor", fs.X_preprocessor), ("estimator", fs.estimator)]
    )

    x = data.drop(["medv"], axis=1, inplace=False)
    y = data[["medv"]]

    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.25)

    with pytest.raises(ValueError) as e:
        _param_mapping(fs.pipeline, x_train, y_train)

    assert str(e.value) == "Invalid JSON Key fake in {}"


def test_core_foreshadow_example_regression():
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_boston
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    import foreshadow as fs

    np.random.seed(0)
    boston = load_boston()
    bostonX_df = pd.DataFrame(boston.data, columns=boston.feature_names)
    bostony_df = pd.DataFrame(boston.target, columns=["target"])
    X_train, X_test, y_train, y_test = train_test_split(
        bostonX_df, bostony_df, test_size=0.2
    )

    model = fs.Foreshadow(estimator=LinearRegression())
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
    import foreshadow as fs

    np.random.seed(0)
    iris = load_iris()
    irisX_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    irisy_df = pd.DataFrame(iris.target, columns=["target"])
    X_train, X_test, y_train, y_test = train_test_split(
        irisX_df, irisy_df, test_size=0.2
    )

    model = fs.Foreshadow(estimator=LogisticRegression())
    model.fit(X_train, y_train)
    score = f1_score(y_test, model.predict(X_test), average="weighted")
    print("Iris score: %f" % score)
