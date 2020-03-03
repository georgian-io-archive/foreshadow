"""Test feature reducer.py"""
import pytest


@pytest.mark.skip("Feature Reducer is not enabled yet.")
def test_feature_reducer_fit_no_ops():
    import numpy as np
    import pandas as pd

    from foreshadow.cachemanager import CacheManager
    from foreshadow.steps import FeatureReducerMapper

    data = pd.DataFrame(
        {
            "age": [10, 20, 33, 44],
            "weights": [20, 30, 50, 60],
            "occupation": ["engineer", "artist", "doctor", "inspector"],
        },
        columns=["age", "weights", "occupation"],
    )
    cs = CacheManager()
    cs["intent", "age"] = "Numeric"
    cs["intent", "weights"] = "Numeric"
    cs["intent", "occupation"] = "Categorical"

    fr = FeatureReducerMapper(cache_manager=cs)
    fr.fit(data)
    transformed_data = fr.transform(data)
    assert np.all(
        np.equal(
            data.values[data.notna()],
            transformed_data.values[transformed_data.notna()],
        )
    )


@pytest.mark.skip("Feature Reducer is not enabled yet.")
def test_feature_reducer_get_mapping_by_intent():
    import pandas as pd

    from foreshadow.cachemanager import CacheManager
    from foreshadow.steps import FeatureReducerMapper
    from foreshadow.steps.preparerstep import PreparerMapping
    from foreshadow.smart import FeatureReducer

    data = pd.DataFrame(
        {
            "age": [10, 20, 33, 44],
            "weights": [20, 30, 50, 60],
            "occupation": ["engineer", "artist", "doctor", "inspector"],
        },
        columns=["age", "weights", "occupation"],
    )
    cs = CacheManager()
    cs["intent", "age"] = "Numeric"
    cs["intent", "weights"] = "Numeric"
    cs["intent", "occupation"] = "Categorical"

    fr = FeatureReducerMapper(cache_manager=cs)
    column_mapping = fr.get_mapping(data)

    check = PreparerMapping()
    check.add(
        ["age", "weights"], [FeatureReducer(cache_manager=cs)], "Numeric"
    )
    check.add(
        ["occupation"], [FeatureReducer(cache_manager=cs)], "Categorical"
    )

    for key in column_mapping.store:
        assert key in check.store
        assert str(column_mapping.store[key]) == str(check.store[key])
