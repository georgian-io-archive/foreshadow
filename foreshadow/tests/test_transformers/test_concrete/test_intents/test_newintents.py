"""Test new intent setup."""

import pytest

from foreshadow.metrics import MetricWrapper


def test_base_intent_get_confidence():
    """Test base intent get_confidence."""

    from foreshadow.intents import BaseIntent

    BaseIntent.confidence_computation = {
        MetricWrapper(lambda x: 1): 0.5,
        MetricWrapper(lambda x: 1): 0.5,
    }

    assert BaseIntent.get_confidence([]) == 1


@pytest.mark.skip("No longer using this type of intent resolving")
def test_intent_ordering_confidence():
    """Test numeric intent get_confidence."""

    import pandas as pd
    import numpy as np

    from foreshadow.intents import Numeric, Categorical, Text

    available_intents = [Numeric, Categorical, Text]
    validation_data = {
        Numeric: pd.DataFrame(np.arange(100)),
        Categorical: pd.DataFrame(["a", "bc", "s", "w", "p"] * 4),
        Text: pd.DataFrame(
            ["hello world", "unit test", "test cases", "reader"]
        ),
    }

    for val_intent, data in validation_data.items():
        scores = {
            sel_intent: sel_intent.get_confidence(data)
            for sel_intent in available_intents
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

    from foreshadow.intents import Categorical

    test = pd.DataFrame(test)

    assert test.equals(Categorical().fit_transform(test))


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
