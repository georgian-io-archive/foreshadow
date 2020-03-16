import pandas as pd
import pytest

from foreshadow.intents import Droppable


def test_droppable_summary():
    df = pd.DataFrame(data=[1, 2, 3])
    with pytest.raises(NotImplementedError):
        Droppable.column_summary(df)
