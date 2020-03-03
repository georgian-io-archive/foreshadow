"""Test preparersteps.py."""

import pytest

from foreshadow.utils.testing import dynamic_import


@pytest.fixture()
def step():
    """Get a PreparerStep subclass instance.

    Note:
        Always returns StandardScaler.

    """
    from foreshadow.steps.preparerstep import PreparerStep
    from foreshadow.steps.autointentmap import AutoIntentMixin

    class Step(PreparerStep, AutoIntentMixin):
        pass

    yield Step


@pytest.mark.parametrize("cache_manager", [True, False])
def test_init_cachemanager(step, cache_manager):
    """Test columnsharer properly init on step.

    Args:
        step: fixture
        cache_manager: True to pass it in, False to not.

    Returns:

    """
    cs = None
    if cache_manager:
        cs = dynamic_import("CacheManager", "foreshadow.cachemanager")()
    step = step(cache_manager=cs)
    assert step.cache_manager is not None
