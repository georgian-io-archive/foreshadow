"""Tests for the foreshadow foreshadow preprocessor step."""

import pytest

from foreshadow.base import BaseEstimator, TransformerMixin
from foreshadow.utils import dynamic_import


@pytest.mark.skip("This is waiting on a patch to the Base class")
def test_preprocessor_none_config(mocker):
    """Tests that a config step can be None.

    Args:
        mocker: A pytest-mocker instance

    """
    import numpy as np
    import pandas as pd
    from foreshadow.cachemanager import CacheManager
    from foreshadow.steps import Preprocessor

    from foreshadow.base import BaseEstimator, TransformerMixin

    class DummyIntent(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None, **fit_params):
            return self

        def transform(self, X, y=None):
            return X

    dummy_config = {
        "cleaner": [],
        "Tiebreak": [DummyIntent],
        "DummyIntent": {"Preprocessor": None},
    }

    mocker.patch(
        "foreshadow.preparer.resolve_config",
        return_value=dummy_config,
        create=True,
    )
    mocker.patch(
        "foreshadow.preparer.Resolver.pick_transformer",
        return_value=DummyIntent(),
        create=True,
    )

    data = pd.DataFrame({"financials": np.arange(10)})
    cs = CacheManager()
    p = Preprocessor(cs)

    p.fit(data)
    _ = p.transform(data)


class DummyIntent(BaseEstimator, TransformerMixin):
    # In order to work with joblib.Parallel, DummyIntent cannot be a local
    # class defined inside a function.
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        return X


@pytest.mark.skip(
    "No longer valid since the mock is now invalid. Need to " "redo"
)
def test_preprocessor_numbers(mocker):
    """Test a standard work flow with preprocessor.

    Args:
        mocker: A pytest-mocker instance

    """
    import numpy as np
    import pandas as pd
    from foreshadow.cachemanager import CacheManager
    from foreshadow.steps import Preprocessor
    from foreshadow.concrete import StandardScaler

    dummy_config = {
        "Cleaner": [],
        "Tiebreak": [DummyIntent],
        "DummyIntent": {"Preprocessor": [StandardScaler]},
    }

    mocker.patch(
        "foreshadow.steps.preprocessor.config.get_config",
        return_value=dummy_config,
        create=True,
    )
    mocker.patch(
        "foreshadow.smart.intent_resolving.intentresolver.IntentResolver"
        ".pick_transformer",
        return_value=DummyIntent(),
        create=True,
    )

    data = pd.DataFrame({"financials": np.arange(10)})
    cs = CacheManager()
    p = Preprocessor(cache_manager=cs)
    p = p.fit(data)
    tf_data = p.transform(data)

    validate = pd.DataFrame(
        {
            "financials": [
                -1.5666989036012806,
                -1.2185435916898848,
                -0.8703882797784892,
                -0.5222329678670935,
                -0.17407765595569785,
                0.17407765595569785,
                0.5222329678670935,
                0.8703882797784892,
                1.2185435916898848,
                1.5666989036012806,
            ]
        }
    )

    assert (tf_data == validate).squeeze().all()


@pytest.mark.skip(
    "No longer valid since the mock is now invalid. Need to " "redo"
)
@pytest.mark.parametrize("cache_manager", [True, False])
def test_preprocessor_cache_manager(mocker, cache_manager):
    """Test a standard work flow with preprocessor with columnsharer.

    Args:
        mocker: A pytest-mocker instance

    """
    import numpy as np
    import pandas as pd
    from foreshadow.steps import Preprocessor

    from foreshadow.concrete import StandardScaler

    dummy_config = {
        "Cleaner": [],
        "Tiebreak": [DummyIntent],
        "DummyIntent": {"Preprocessor": [StandardScaler]},
    }

    mocker.patch(
        "foreshadow.steps.preprocessor.config.get_config",
        return_value=dummy_config,
        create=True,
    )
    mocker.patch(
        "foreshadow.smart.intent_resolving.intentresolver.IntentResolver"
        ".pick_transformer",
        return_value=DummyIntent(),
        create=True,
    )
    cs = None
    if cache_manager:
        cs = dynamic_import("CacheManager", "foreshadow.cachemanager")()

    data = pd.DataFrame({"financials": np.arange(10)})
    p = Preprocessor(cache_manager=cs)
    p = p.fit(data)
    tf_data = p.transform(data)

    validate = pd.DataFrame(
        {
            "financials": [
                -1.5666989036012806,
                -1.2185435916898848,
                -0.8703882797784892,
                -0.5222329678670935,
                -0.17407765595569785,
                0.17407765595569785,
                0.5222329678670935,
                0.8703882797784892,
                1.2185435916898848,
                1.5666989036012806,
            ]
        }
    )

    assert (tf_data == validate).squeeze().all()
