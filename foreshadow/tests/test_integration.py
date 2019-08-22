"""Integration tests.

Slow-running tests that verify the performance of the framework on simple
datasets.

"""
import pytest


def check_slow():
    import os

    os.environ["FORESHADOW_TESTS"] = "ALL"
    return os.environ.get("FORESHADOW_TESTS") != "ALL"


slow = pytest.mark.skipif(
    check_slow(), reason="Skipping long-runnning integration tests"
)


@slow
def test_integration_binary_classification():
    import foreshadow as fs
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
    shadow = fs.Foreshadow(estimator=LogisticRegression())
    shadow.fit(X_train, y_train)

    baseline = 0.9824561403508771
    score = shadow.score(X_test, y_test)

    assert not score < baseline * 0.9


@slow
def test_integration_binary_classification_with_optimization():
    import foreshadow as fs
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from foreshadow.optimizers import RandomSearchCV

    np.random.seed(1337)

    cancer = load_breast_cancer()
    cancerX_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    cancery_df = pd.DataFrame(cancer.target, columns=["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        cancerX_df, cancery_df, test_size=0.2
    )
    ok = {
        "iid": True,
        "scoring": "accuracy",
        "n_iter": 15,
        "return_train_score": True,
    }
    shadow = fs.Foreshadow(
        estimator=LogisticRegression(),
        optimizer=RandomSearchCV,
        optimizer_kwargs=ok,
    )
    shadow.fit(X_train, y_train)

    baseline = 0.9824561403508771
    score = shadow.score(X_test, y_test)

    assert not score < baseline * 0.9


@slow
def test_integration_multiclass_classification():
    import foreshadow as fs
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    np.random.seed(1337)

    iris = load_iris()
    irisX_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    irisy_df = pd.DataFrame(iris.target, columns=["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        irisX_df, irisy_df, test_size=0.2
    )
    shadow = fs.Foreshadow(estimator=LogisticRegression())
    shadow.fit(X_train, y_train)

    baseline = 0.9666666666666667
    score = shadow.score(X_test, y_test)

    assert not score < baseline * 0.9


@slow
def test_integration_regression():
    import foreshadow as fs
    import pandas as pd
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    boston = load_boston()
    bostonX_df = pd.DataFrame(boston.data, columns=boston.feature_names)
    bostony_df = pd.DataFrame(boston.target, columns=["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        bostonX_df, bostony_df, test_size=0.2
    )
    shadow = fs.Foreshadow(estimator=LinearRegression())
    shadow.fit(X_train, y_train)

    baseline = 0.6953024611269096
    score = shadow.score(X_test, y_test)

    assert not score < baseline * 0.9
