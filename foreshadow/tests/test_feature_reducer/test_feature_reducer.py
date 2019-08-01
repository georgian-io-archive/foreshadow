"""Test feature reducer.py"""
import numpy as np
import pandas as pd

from foreshadow.core.column_sharer import ColumnSharer
from foreshadow.feature_reducer import FeatureReducer, SmartFeatureReducer


def test_feature_reducer_fit_no_ops():
    data = pd.DataFrame(
        {
            "age": [10, 20, 33, 44],
            "weights": [20, 30, 50, 60],
            "financials": ["$1.00", "$550.01", "$1234", "$12353.3345"],
        },
        columns=["age", "weights", "financials"],
    )
    cs = ColumnSharer()
    fr = FeatureReducer(cs)
    fr.fit(data)
    transformed_data = fr.transform(data)
    assert np.all(
        np.equal(
            data.values[data.notna()],
            transformed_data.values[transformed_data.notna()],
        )
    )


def test_feature_reducer_get_mapping_by_intent():
    data = pd.DataFrame(
        {
            "age": [10, 20, 33, 44],
            "weights": [20, 30, 50, 60],
            "occupation": ["engineer", "artist", "doctor", "inspector"],
        },
        columns=["age", "weights", "occupation"],
    )
    cs = ColumnSharer()
    cs["intent", "age"] = "Numeric"
    cs["intent", "weights"] = "Numeric"
    cs["intent", "occupation"] = "Categorical"

    fr = FeatureReducer(cs)
    column_mapping = fr.get_mapping(data)
    check = {
        0: {"inputs": (["age", "weights"],), "steps": [SmartFeatureReducer()]},
        1: {"inputs": (["occupation"],), "steps": [SmartFeatureReducer()]},
    }
    assert str(column_mapping) == str(check)
