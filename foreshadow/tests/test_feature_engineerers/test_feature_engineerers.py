"""Test feature_engineerer.py"""
import pandas as pd

from foreshadow.core.column_sharer import ColumnSharer
from foreshadow.feature_engineerers import (
    FeatureEngineerer,
    SmartFeatureEngineerer,
)


def test_feature_engineerer_fit():
    """Test basic fit call."""
    import numpy as np

    data = pd.DataFrame(
        {
            "age": [10, 20, 33, 44],
            "weights": [20, 30, 50, 60],
            "financials": ["$1.00", "$550.01", "$1234", "$12353.3345"],
        },
        columns=["age", "weights", "financials"],
    )
    cs = ColumnSharer()
    cs["domain"]["age"] = "personal"
    cs["domain"]["weights"] = "personal"
    cs["domain"]["financials"] = "financial"

    cs["intent"]["age"] = "NumericIntent"
    cs["intent"]["weights"] = "NumericIntent"
    cs["intent"]["financials"] = "FinancialIntent"

    dc = FeatureEngineerer(cs)
    dc.fit(data)
    data = dc.transform(data)
    check = pd.DataFrame(
        {
            "age": [10, 20, 33, 44],
            "weights": [20, 30, 50, 60],
            "financials": ["$1.00", "$550.01", "$1234", "$12353.3345"],
        },
        columns=["age", "weights", "financials"],
    )
    assert np.all(
        np.equal(data.values[data.notna()], check.values[check.notna()])
    )


def test_feature_engineerer_get_mapping():
    """Test basic fit call."""

    data = pd.DataFrame(
        {
            "age": [10, 20, 33, 44],
            "weights": [20, 30, 50, 60],
            "financials": ["$1.00", "$550.01", "$1234", "$12353.3345"],
        },
        columns=["age", "weights", "financials"],
    )
    cs = ColumnSharer()
    cs["domain"]["age"] = "personal"
    cs["domain"]["weights"] = "personal"
    cs["domain"]["financials"] = "financial"

    cs["intent"]["age"] = "NumericIntent"
    cs["intent"]["weights"] = "NumericIntent"
    cs["intent"]["financials"] = "FinancialIntent"

    dc = FeatureEngineerer(cs)
    column_mapping = dc.get_mapping(data)
    check = {
        0: {
            "inputs": (["age", "weights"],),
            "steps": [SmartFeatureEngineerer()],
        },
        1: {"inputs": (["financials"],), "steps": [SmartFeatureEngineerer()]},
    }
    assert str(column_mapping) == str(check)
