"""Test intent resolution steps."""


def test_resolver_overall():
    """Big picture intent resolution test."""

    import numpy as np
    import pandas as pd
    from foreshadow.core.column_sharer import ColumnSharer
    from foreshadow.core.preparersteps.resolver import IntentResolver

    columns = ["financials"]
    data = pd.DataFrame({"financials": np.arange(100)}, columns=columns)
    cs = ColumnSharer()
    ir = IntentResolver(cs)
    ir.fit(data)

    assert cs["intent", "financials"] == "Numeric"
