"""Test random_search.py"""

import pytest


@pytest.fixture()
def simple_distribution():
    """Simple parameter distribution for testing."""
    from foreshadow.optimizers import ParamSpec

    ps = ParamSpec()
    dist = [
        {
            "s__transformer": "StandardScaler",
            "s__transformer__with_mean": [False, True],
        },
        {
            "s__transformer": "MinMaxScaler",
            "s__transformer__feature_range": [(0, 1), (0, 0.5)],
        },
    ]
    ps.set_params(**{"param_distributions": dist})
    yield ps


@pytest.fixture()
def iris_data():
    """Iris dataset."""
    import sklearn.datasets as dt
    import pandas as pd

    data = dt.load_iris()
    X_data = pd.DataFrame(data.data, columns=data.feature_names).iloc[:, 0]
    y_data = pd.DataFrame(data.target, columns=["target"])["target"]
    return X_data, y_data


@pytest.fixture()
def estimator_counter(mocker):
    """Mocked estimator. .keys method must be set to return all possible keys
    from the parameter distribution."""
    counter = []

    class Estimator:
        def __init__(self, **kwargs):
            pass

        def set_params(self, *args, **kwargs):
            counter.append(kwargs)
            return self

        def get_params(self, deep=True):
            return self.keys()

    Estimator.fit = mocker.Mock(return_value=None)
    Estimator.score = mocker.Mock(return_value=0.5)
    return Estimator, counter


@pytest.mark.skip(
    "Due to upgrade, the base search class in sklearn has "
    "changed. Since we are not using CV research yet, "
    "I'm turning it off."
)
def test_random_search_simple(
    estimator_counter, simple_distribution, iris_data
):
    """Test that random search finds all different parameter specifications.

    Args:
        estimator_counter: fixture estimator_counter
        simple_distribution: fixture distribution to parameter optimize on.
        iris_data: fixture dataset to use.

    """
    from foreshadow.optimizers import RandomSearchCV

    estimator, counter = estimator_counter
    dist = simple_distribution
    keys = {key: None for d in dist.param_distributions for key in d}
    estimator.keys = lambda x: keys
    estimator = estimator()
    X, y = iris_data
    rs = RandomSearchCV(estimator=estimator, param_distributions=dist)
    rs.fit(X, y)
    unique_samples = set()
    for sample in counter:
        v = ""
        for val in sample.values():
            v += str(val)
        unique_samples.add(v)

    assert len(unique_samples) == 4


@pytest.mark.skip("Temporarily turning it off since the feature is not ready")
def test_random_param_list_simple(simple_distribution):
    """Test that sampler properly iterates over parameter distribution.

    Args:
        simple_distribution: fixture parameter distribution.

    Returns:

    """
    from foreshadow.optimizers.random_search import HyperOptRandomSampler

    dist = simple_distribution
    Sampler = HyperOptRandomSampler(dist, 10, max_tries=100)
    samples = []
    for sample in Sampler:
        samples.append(sample)
    unique_samples = set()
    for sample in samples:
        v = ""
        for val in sample.values():
            v += str(val)
        unique_samples.add(v)
    print(unique_samples)
    assert len(unique_samples) == 4  # 4 unique samples.


@pytest.mark.skip("Temporarily turning it off since the feature is not ready")
def test_random_param_list_simple_non_unique(simple_distribution):
    """Test that sampler properly gives non unique iterations.

    Args:
        simple_distribution: fixture parameter distribution.

    Returns:

    """
    from foreshadow.optimizers.random_search import HyperOptRandomSampler

    dist = simple_distribution
    Sampler = HyperOptRandomSampler(dist, 10, max_tries=None)
    assert len(Sampler) == 10  # 10 non unique samples.
