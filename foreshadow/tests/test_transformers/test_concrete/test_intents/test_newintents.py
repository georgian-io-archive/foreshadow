"""Test new intent setup."""

import pytest


def test_base_intent_get_confidence():
    """Test base intent get_confidence."""

    from foreshadow.intents import BaseIntent

    BaseIntent.confidence_computation = {
        (lambda x: 1): 0.5,
        (lambda x: 1): 0.5,
    }

    assert BaseIntent.get_confidence([]) == 1


def test_intent_ordering_confidence():
    """Test numeric intent get_confidence."""

    import pandas as pd
    import numpy as np

    from foreshadow.intents import Numeric, Categoric, Text

    validation_data = {
        Numeric: pd.DataFrame(np.arange(100)),
        Categoric: pd.DataFrame([1, 2, 3, 4, 5] * 4),
        Text: pd.DataFrame(["hello", "unit", "test", "reader"]),
    }

    for val_intent, data in validation_data.items():
        scores = {
            sel_intent: sel_intent.get_confidence(data)
            for sel_intent in validation_data.keys()
        }
        assert max(scores, key=scores.get) == val_intent


@pytest.mark.parametrize(
    "test,val", [([1, 2, 3], [1, 2, 3]), ([1, "test", 10], [1, None, 10])]
)
def test_intent_numeric_transform(test, val):
    """Test numeric data is transformed.

    Args:
        test: parametrized arguments (test, validation)

    """

    import pandas as pd

    from foreshadow.intents import Numeric

    test = pd.DataFrame(test)
    val = pd.DataFrame(val)

    assert val.equals(Numeric().fit_transform(test))


@pytest.mark.parametrize("test", [([1, 2, 3]), (["hello", "test", "world"])])
def test_intent_categoric_transform(test):
    """Test numeric data is transformed.

    Args:
        test: parametrized arguments (input == output)

    """

    import pandas as pd

    from foreshadow.intents import Categoric

    test = pd.DataFrame(test)

    assert test.equals(Categoric().fit_transform(test))


@pytest.mark.parametrize(
    "test,val",
    [([1, 2, 3], ["1", "2", "3"]), ([1, "test", 10], ["1", "test", "10"])],
)
def test_intent_text_transform(test, val):
    """Test numeric data is transformed.

    Args:
        test: parametrized arguments (test, validation)

    """

    import pandas as pd

    from foreshadow.intents import Text

    test = pd.DataFrame(test)
    val = pd.DataFrame(val)

    assert val.equals(Text().fit_transform(test))