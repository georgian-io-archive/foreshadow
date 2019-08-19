"""Test random_search.py"""

import pytest


@pytest.fixture()
def simple_distribution():
    yield [
            {
                "s__transformer": "StandardScaler",
                "s__transformer__with_mean": [False, True],
            },
            {
                "s__transformer": "MinMaxScaler",
                "s__transformer__feature_range": [(0, 1), (0, 0.5)]
            },
        ]


def test_random_search_simple(simple_distribution):
    dist = simple_distribution


def test_random_param_list_simple(simple_distribution):
    from foreshadow.optimizers.random_search import HyperOptRandomSampler
    from foreshadow.optimizers import ParamSpec
    dist = simple_distribution
    ps = ParamSpec()
    ps.set_params(**{"param_distributions": dist})
    Sampler = HyperOptRandomSampler(ps, 10)
    samples = []
    for sample in Sampler:
        samples.append(sample)
    assert len(samples) == 4  # 4 unique samples.
