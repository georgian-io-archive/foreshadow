import pandas as pd

from foreshadow.intents import Droppable


def test_droppable_summary():
    df = pd.DataFrame(data=[1, 2, 3])
    summary = Droppable.column_summary(df)
    assert "count" in summary
