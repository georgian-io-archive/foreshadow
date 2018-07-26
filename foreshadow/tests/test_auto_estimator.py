import itertools

import pytest


def test_auto_estimator_config_get_tpot_config():
    from ..estimators.config import get_tpot_config

    setup1 = get_tpot_config("classification", include_preprocessors=True)
    setup2 = get_tpot_config("regression", include_preprocessors=True)
    setup3 = get_tpot_config("classification")
    setup4 = get_tpot_config("regression")
    check_setup_keys = lambda x: not any(
        k in x.keys() for k in ["estimators", "preprocessors"]
    )

    assert (
        check_setup_keys(setup1)
        and check_setup_keys(setup2)
        and (
            check_setup_keys(setup3)
            and all(k in setup1.keys() for k in setup3)
            and setup1 != setup3
        )
        and (
            check_setup_keys(setup4)
            and all(k in setup2.keys() for k in setup4)
            and setup2 != setup4
        )
    )


def test_invalid_problem_type():
    from ..estimators.auto_estimator import AutoEstimator

    with pytest.raises(ValueError) as e:
        ae = AutoEstimator(problem_type="test")
    assert "problem type must be in " in str(e.value)


def test_invalid_auto_estimator():
    from ..estimators.auto_estimator import AutoEstimator

    with pytest.raises(ValueError) as e:
        ae = AutoEstimator(auto_estimator="test")
    assert "auto_estimator must be in " in str(e.value)


def test_invalid_kwargs_vague_estimator():
    from ..estimators.auto_estimator import AutoEstimator

    with pytest.raises(ValueError) as e:
        ae = AutoEstimator(estimator_kwargs="test")
    assert (
        str(e.value)
        == "estimator_kwargs can only be set when estimator and problem are specified"
    )


def test_invalid_kwargs_not_dict():
    from ..estimators.auto_estimator import AutoEstimator

    with pytest.raises(ValueError) as e:
        ae = AutoEstimator(
            problem_type="regression", auto_estimator="tpot", estimator_kwargs="test"
        )
    assert str(e.value) == "estimator_kwargs must be a valid kwarg dictionary"


@pytest.mark.parametrize(
    "problem_type,auto_estimator",
    list(itertools.product(["regression", "classification"], ["tpot", "autosklearn"])),
)
def test_invalid_kwarg_dict(problem_type, auto_estimator):
    from ..estimators.auto_estimator import AutoEstimator

    with pytest.raises(ValueError) as e:
        ae = AutoEstimator(
            problem_type=problem_type,
            auto_estimator=auto_estimator,
            estimator_kwargs={"test": "test"},
        )
    assert "The following invalid kwargs were passed in:" in str(e.value)


def test_default_estimator_setup_classification():
    import numpy as np
    import pandas as pd
    from autosklearn.classification import AutoSklearnClassifier

    from ..estimators.auto_estimator import AutoEstimator

    y = pd.DataFrame(np.array([0] * 50 + [1] * 50))
    ae = AutoEstimator()
    est = ae._setup_estimator(y)
    assert isinstance(est, AutoSklearnClassifier)


def test_default_estimator_setup_regression():
    import numpy as np
    import pandas as pd
    from tpot import TPOTRegressor

    from ..estimators.auto_estimator import AutoEstimator

    y = pd.DataFrame(np.random.normal(0, 1, 200))
    ae = AutoEstimator()
    est = ae._setup_estimator(y)
    assert isinstance(est, TPOTRegressor)


@pytest.mark.skip(reason="Waiting on issue response @github:automl/autosklearn")
@pytest.mark.slowest
def test_auto_estimator_default_to_autosklearn():
    import random
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from ..estimators.auto_estimator import AutoEstimator

    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    X = pd.DataFrame(np.array([0] * 50 + [1] * 50).reshape((-1, 1)))
    y = pd.DataFrame(np.array([0] * 50 + [1] * 50))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    ae = AutoEstimator(
        problem_type="classification",
        auto_estimator="autosklearn",
        estimator_kwargs={"time_left_for_this_task": 20, "seed": seed},
    )
    ae.fit(X, y)
    ae_predict = ae.predict(X_test)
    ae_predict_proba = ae.predict_proba(X_test)
    ae_score = ae.score(X_test, y_test)
    expected_predict = np.array([0, 1, 0, 1, 1, 1, 0, 1, 1, 1])
    expected_predict_proba = np.array(
        [
            [0.8584763163857105, 0.14152368227318532],
            [0.13621543275812661, 0.8637845659007688],
            [0.8584763163857105, 0.14152368227318532],
            [0.13621543275812661, 0.8637845659007688],
            [0.13621543275812661, 0.8637845659007688],
            [0.13621543275812661, 0.8637845659007688],
            [0.8584763163857105, 0.14152368227318532],
            [0.13621543275812661, 0.8637845659007688],
            [0.1362179604041567, 0.863782038254739],
            [0.1362179604041567, 0.863782038254739],
        ]
    )
    expected_score = 1.0

    print(ae_predict.tolist())
    print(ae_predict_proba.tolist())
    print(ae_predict_score)

    raise Exception()

    assert np.allclose(ae_predict, expected_predict)
    assert np.allclose(ae_predict_proba, expected_predict_proba)
    assert np.allclose(ae_score, expected_score)


@pytest.mark.slow
def test_auto_estimator_default_to_tpot():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from ..estimators.auto_estimator import AutoEstimator

    seed = 0
    np.random.seed(seed)
    X = pd.DataFrame(np.arange(200).reshape((-1, 1)))
    y = pd.DataFrame(np.random.normal(0, 1, 200))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    ae = AutoEstimator(
        problem_type="regression",
        auto_estimator="tpot",
        estimator_kwargs={"generations": 1, "population_size": 5, "random_state": seed},
    )
    ae.fit(X, y)
    ae_predict = ae.predict(X_test)
    ae_score = ae.score(X_test, y_test)

    expected_predict = np.array(
        [
            0.07091049314116117,
            0.07091049314116117,
            0.07091049314116117,
            0.07091049314116117,
            0.07091049314116117,
            0.07091049314116117,
            0.07091049314116117,
            0.07091049314116117,
            0.07091049314116117,
            0.07091049314116117,
            0.07091049314116117,
            0.07091049314116117,
            0.07091049314116117,
            0.07091049314116117,
            0.07091049314116117,
            0.07091049314116117,
            0.07091049314116117,
            0.07091049314116117,
            0.07091049314116117,
            0.07091049314116117,
        ]
    )
    expected_score = -0.8640193307896562

    assert np.allclose(ae_predict, expected_predict)
    assert np.allclose(ae_score, expected_score)
