import pytest


def test_generic_intent_is_intent():
    import pandas as pd
    from foreshadow.intents import GenericIntent

    X = pd.DataFrame([1, 2, 3])

    assert GenericIntent.is_intent(X)


def test_numeric_intent_is_intent():
    import pandas as pd
    from foreshadow.intents import NumericIntent

    X = pd.DataFrame([1, 2, 3])
    X1 = pd.DataFrame([1, 2, "Test"])

    assert NumericIntent.is_intent(X)
    assert NumericIntent.is_intent(X1)


def test_categorical_intent_is_intent_numeric():
    import pandas as pd
    from foreshadow.intents import CategoricalIntent

    X = pd.DataFrame([1] * 10 + [2] * 20)
    X1 = pd.DataFrame(list(range(0, 100)))

    assert CategoricalIntent.is_intent(X)
    assert not CategoricalIntent.is_intent(X1)


def test_categorical_intent_is_intent_string():
    import pandas as pd
    from foreshadow.intents import CategoricalIntent

    X = pd.DataFrame(["test"] * 10 + ["hi"] * 10)

    assert CategoricalIntent.is_intent(X)
