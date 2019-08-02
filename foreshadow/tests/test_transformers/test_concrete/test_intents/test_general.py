import pytest


@pytest.mark.skip("to be removed and replaced with new intent equiv.")
def test_generic_intent_is_intent():
    import pandas as pd
    from foreshadow.concrete import GenericIntent

    X = pd.DataFrame([1, 2, 3])

    assert GenericIntent.is_intent(X)


@pytest.mark.skip("to be removed and replaced with new intent equiv.")
def test_generic_intent_column_summary():
    import pandas as pd
    from foreshadow.concrete import GenericIntent

    X = pd.DataFrame([1, 2, 3])

    assert not GenericIntent.column_summary(X)


@pytest.mark.skip("to be removed and replaced with new intent equiv.")
def test_numeric_intent_is_intent():
    import pandas as pd
    from foreshadow.concrete import NumericIntent

    X = pd.DataFrame([1, 2, 3])
    X1 = pd.DataFrame([1, 2, "Test"])

    assert NumericIntent.is_intent(X)
    assert NumericIntent.is_intent(X1)


@pytest.mark.skip("to be removed and replaced with new intent equiv.")
def test_mode_freq():
    import numpy as np
    import pandas as pd
    from foreshadow.concrete import _mode_freq

    np.random.seed(0)
    X1 = pd.Series([])
    X2 = pd.Series([0])
    X3 = pd.Series([0, 10, 10])
    X4 = pd.Series([0, 10, 10, 20, 20])
    X5 = pd.Series(np.random.randint(0, 10, 100))

    assert _mode_freq(X1) == ([], [])
    assert _mode_freq(X2) == ([0], [[0, 1, 1.0]])
    assert _mode_freq(X3) == ([10], [[10, 2, 2 / 3], [0, 1, 1 / 3]])
    assert _mode_freq(X4) == (
        [10, 20],
        [[20, 2, 0.4], [10, 2, 0.4], [0, 1, 0.2]],
    )
    assert _mode_freq(X5) == (
        [3],
        [
            [3, 15, 0.15],
            [4, 14, 0.14],
            [5, 11, 0.11],
            [0, 11, 0.11],
            [9, 10, 0.1],
            [7, 10, 0.1],
            [1, 9, 0.09],
            [8, 8, 0.08],
            [2, 7, 0.07],
            [6, 5, 0.05],
        ],
    )


@pytest.mark.skip("to be removed and replaced with new intent equiv.")
def test_outliers():
    import numpy as np
    import pandas as pd
    from foreshadow.concrete import _outliers

    np.random.seed(0)
    X = pd.Series(
        np.concatenate(
            [
                np.random.randint(0, 10, 100),
                np.random.randint(-1000, -900, 5),
                np.random.randint(900, 1000, 5),
            ]
        )
    )
    expected_arr = np.array(
        [-997, 977, 973, -958, -952, 921, 910, -907, -902, 900]
    )
    assert np.array_equal(_outliers(X).values, expected_arr)


@pytest.mark.skip("to be removed and replaced with new intent equiv.")
def test_numeric_intent_column_summary():
    import numpy as np
    import pandas as pd
    from foreshadow.concrete import NumericIntent

    np.random.seed(0)
    X = pd.DataFrame(
        np.concatenate(
            [
                np.rint(np.random.normal(50, 10, 1000)),
                np.random.randint(100, 200, 10),
            ]
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
            [54.0, 49.0, 0.048514851485148516],
            [46.0, 49.0, 0.048514851485148516],
            [49.0, 47.0, 0.046534653465346534],
            [50.0, 42.0, 0.041584158415841586],
            [45.0, 41.0, 0.040594059405940595],
            [52.0, 41.0, 0.040594059405940595],
            [43.0, 40.0, 0.039603960396039604],
            [53.0, 39.0, 0.03861386138613861],
            [48.0, 36.0, 0.03564356435643564],
            [41.0, 36.0, 0.03564356435643564],
        ],
    }

    assert NumericIntent.column_summary(X) == expected_dict


@pytest.mark.skip("to be removed and replaced with new intent equiv.")
def test_categorical_intent_is_intent_numeric():
    import pandas as pd
    from foreshadow.concrete import CategoricalIntent

    X = pd.DataFrame([1] * 10 + [2] * 20)
    X1 = pd.DataFrame(list(range(0, 100)))

    assert CategoricalIntent.is_intent(X)
    assert not CategoricalIntent.is_intent(X1)


@pytest.mark.skip("to be removed and replaced with new intent equiv.")
def test_categorical_intent_is_intent_string():
    import pandas as pd
    from foreshadow.concrete import CategoricalIntent

    X = pd.DataFrame(["test"] * 10 + ["hi"] * 10)

    assert CategoricalIntent.is_intent(X)


@pytest.mark.skip("to be removed and replaced with new intent equiv.")
def test_standard_intent_column_summary():
    import numpy as np
    import pandas as pd
    from foreshadow.concrete import _standard_col_summary

    X = pd.DataFrame(["test"] * 5 + ["hi"] * 10 + [np.nan] * 5)
    expected_dict = {
        "mode": ["hi"],
        "nan": 5,
        "top10": [["hi", 10, 0.5], ["test", 5, 0.25]],
    }

    assert _standard_col_summary(X) == expected_dict


@pytest.mark.skip("to be removed and replaced with new intent equiv.")
def test_standard_intent_column_summary_calls():
    import numpy as np
    import pandas as pd
    from foreshadow.concrete import CategoricalIntent, TextIntent

    X = pd.DataFrame(["test"] * 5 + ["hi"] * 10 + [np.nan] * 5)

    CategoricalIntent.column_summary(X)
    TextIntent.column_summary(X)
