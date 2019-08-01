"""Test intent resolution steps."""


def test_resolver_overall():
    """Big picture intent resolution test."""

    import numpy as np
    import pandas as pd
    from foreshadow.preparer.column_sharer import ColumnSharer
    from foreshadow.preparer.steps.resolver import ResolverMapper

    columns = ["financials"]
    data = pd.DataFrame({"financials": np.arange(100)}, columns=columns)
    cs = ColumnSharer()
    ir = ResolverMapper(cs)
    ir.fit(data)

    assert cs["intent", "financials"] == "Numeric"
