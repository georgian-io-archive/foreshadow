import pytest


def test_generic_intent_is_intent():
    import pandas as pd
    from foreshadow.intents import GenericIntent

    X = pd.DataFrame([1, 2, 3])

    assert GenericIntent.is_intent(X)


def test_generic_intent_column_summary():
    import pandas as pd
    from foreshadow.intents import GenericIntent

    X = pd.DataFrame([1, 2, 3])

    assert not GenericIntent.column_summary(X)


def test_numeric_intent_is_intent():
    import pandas as pd
    from foreshadow.intents import NumericIntent

    X = pd.DataFrame([1, 2, 3])
    X1 = pd.DataFrame([1, 2, "Test"])

    assert NumericIntent.is_intent(X)
    assert NumericIntent.is_intent(X1)


def test_mode_freq():
    import numpy as np
    import pandas as pd
    from foreshadow.intents import mode_freq

    np.random.seed(0)
    X0 = pd.Series([0])
    X_ = pd.Series([0, 10, 10])
    X1 = pd.Series([0, 10, 10, 20, 20])
    X2 = pd.Series(np.random.randint(0, 10, 100))

    assert mode_freq(X0) == (None, [])
    assert mode_freq(X_) == (10, [[10, 2], [0, 1]])
    assert mode_freq(X1) == ([10, 20], [[20, 2], [10, 2], [0, 1]])
    assert mode_freq(X2) == (
        3,
        [
            [3, 15],
            [4, 14],
            [5, 11],
            [0, 11],
            [9, 10],
            [7, 10],
            [1, 9],
            [8, 8],
            [2, 7],
            [6, 5],
        ],
    )


def test_numeric_intent_column_summary():
    import numpy as np
    import pandas as pd
    from foreshadow.intents import NumericIntent

    np.random.seed(0)
    X = pd.DataFrame(
        np.concatenate(
            [np.rint(np.random.normal(50, 10, 1000)), np.random.randint(100, 200, 10)]
        )
    )
    expected_dict = {
        "10outliers": [
            197.0,
            196.0,
            191.0,
            161.0,
            157.0,
            141.0,
            138.0,
            131.0,
            122.0,
            121.0,
        ],
        "25th": 43.0,
        "75th": 56.0,
        "invalid": 0,
        "max": 197.0,
        "mean": 50.5960396039604,
        "median": 49.0,
        "min": 20.0,
        "mode": [46.0, 54.0],
        "nan": 0,
        "std": 14.658355220098485,
        "top10": [
            [54.0, 49.0],
            [46.0, 49.0],
            [49.0, 47.0],
            [50.0, 42.0],
            [45.0, 41.0],
            [52.0, 41.0],
            [43.0, 40.0],
            [53.0, 39.0],
            [48.0, 36.0],
            [41.0, 36.0],
        ],
    }

    assert NumericIntent.column_summary(X) == expected_dict


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


def test_categorical_intent_column_summary():
    import numpy as np
    import pandas as pd
    from foreshadow.intents import CategoricalIntent

    X = pd.DataFrame(["test"] * 5 + ["hi"] * 10 + [np.nan] * 5)
    expected_dict = {"mode": "hi", "nan": 5, "top10": [["hi", 10], ["test", 5]]}

    assert CategoricalIntent.column_summary(X) == expected_dict
