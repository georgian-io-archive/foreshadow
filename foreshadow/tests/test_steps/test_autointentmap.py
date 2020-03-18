"""Test autointentmap.py"""

import pytest


@pytest.fixture()
def step():
    """Get a PreparerStep subclass instance.

    Note:
        Always returns StandardScaler.

    """
    from foreshadow.steps.preparerstep import PreparerStep
    from foreshadow.steps.autointentmap import AutoIntentMixin
    from foreshadow.cachemanager import CacheManager

    class Step(PreparerStep, AutoIntentMixin):
        def get_mapping(self, X):
            self.check_resolve(X)

    yield Step(cache_manager=CacheManager())


def test_autointentmapping(step):
    """Test intents automatically mapped for a PreparerStep subclass."""
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(
        [np.arange(i, i + 2) for i in range(100)], columns=["1", "2"]
    )
    step.get_mapping(df)
    assert step.cache_manager["intent", "1"] == "Droppable"
    assert step.cache_manager["intent", "2"] == "Droppable"
