"""Test intent resolution steps."""


def test_resolver_overall():
    """Big picture intent resolution test."""

    import numpy as np
    import pandas as pd
    from foreshadow.columnsharer import ColumnSharer
    from foreshadow.steps import IntentMapper

    columns = ["financials"]
    data = pd.DataFrame({"financials": np.arange(100)}, columns=columns)
    cs = ColumnSharer()
    ir = IntentMapper(column_sharer=cs)
    ir.fit(data)
    assert cs["intent", "financials"] == "Numeric"
