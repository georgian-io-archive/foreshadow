"""Test intent resolution steps."""


def test_resolver_overall():
    """Big picture intent resolution test."""

    import numpy as np
    import pandas as pd
    from foreshadow.cachemanager import CacheManager
    from foreshadow.steps import IntentMapper

    columns = ["financials"]
    data = pd.DataFrame({"financials": np.arange(100)}, columns=columns)
    cs = CacheManager()
    ir = IntentMapper(cache_manager=cs)
    ir.fit(data)
    assert cs["intent", "financials"] == "Droppable"
