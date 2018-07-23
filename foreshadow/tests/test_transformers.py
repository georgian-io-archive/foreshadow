import pytest


def test_transformer_wrapper_init():

    from ..transformers import StandardScaler

    scaler = StandardScaler(name="test-scaler", keep_columns=True)

    assert scaler.name == "test-scaler"
    assert scaler.keep_columns == True


def test_transformer_wrapper_function():

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler as StandardScaler
    from ..transformers import StandardScaler as CustomScaler

    df = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    custom = CustomScaler()
    sklearn = StandardScaler()

    custom.fit(df[["crim"]])
    sklearn.fit(df[["crim"]])

    custom_tf = custom.transform(df[["crim"]])
    sklearn_tf = custom.transform(df[["crim"]])

    assert np.array_equal(custom_tf.values, sklearn_tf)

    custom_tf = custom.fit_transform(df[["crim"]])
    sklearn_tf = sklearn.fit_transform(df[["crim"]])

    assert np.array_equal(custom_tf.values, sklearn_tf)


def test_transformer_naming_override():

    from ..transformers import StandardScaler
    import pandas as pd

    df = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    scaler = StandardScaler(name="test", keep_columns=False)
    out = scaler.fit_transform(df[["crim"]])

    assert out.iloc[:, 0].name == "crim_test_0"


def test_transformer_naming_default():

    from ..transformers import StandardScaler
    import pandas as pd

    df = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    scaler = StandardScaler(keep_columns=False)
    out = scaler.fit_transform(df[["crim"]])

    assert out.iloc[:, 0].name == "crim_StandardScaler_0"


def test_transformer_arallel_invalid():

    import pandas as pd

    from ..transformers import ParallelProcessor

    df = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    class InvalidTransformer:
        pass

    t = InvalidTransformer()

    with pytest.raises(TypeError) as e:
        proc = ParallelProcessor([("scaled", ["crim", "zn", "indus"], t)])

    assert str(e.value) == (
        "All estimators should implement fit and "
        "transform. '{}'"
        " (type {}) doesn't".format(t, type(t))
    )


def test_transformer_parallel_empty():

    import pandas as pd
    import numpy as np

    from ..transformers import ParallelProcessor

    df = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    proc = ParallelProcessor([("scaled", ["crim", "zn", "indus"], None)])

    proc.fit(df)
    tf = proc.transform(df)

    assert tf.equals(df[[]])

    tf = proc.fit_transform(df)

    assert tf.equals(df[[]])


def test_transformer_parallel():

    import pandas as pd

    from ..transformers import ParallelProcessor
    from ..transformers import StandardScaler

    df = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    ss = StandardScaler(name="scaled")

    proc = ParallelProcessor(
        [("scaled", ["crim", "zn", "indus"], StandardScaler(keep_columns=False))]
    )

    ss.fit(df[["crim", "zn", "indus"]])
    proc.fit(df)

    tf = proc.transform(df)
    tf_norm = ss.transform(df[["crim", "zn", "indus"]])

    assert tf.equals(tf_norm)


def test_transformer_pipeline():

    import pandas as pd
    import numpy as np

    np.random.seed(1337)

    from ..transformers import StandardScaler as CustomScaler
    from ..transformers import ParallelProcessor

    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import FeatureUnion

    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression

    df = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    target = df["medv"]
    df = df[["crim", "zn", "indus"]]
    test = df.copy(deep=True)

    custom = Pipeline(
        [
            (
                "Step1",
                ParallelProcessor(
                    [
                        (
                            "scaled",
                            ["crim", "zn", "indus"],
                            CustomScaler(keep_columns=False),
                        )
                    ]
                ),
            ),
            ("estimator", LinearRegression()),
        ]
    )

    sklearn = Pipeline(
        [
            ("Step1", FeatureUnion([("scaled", StandardScaler())])),
            ("estimator", LinearRegression()),
        ]
    )

    sklearn.fit(df, target)
    custom.fit(df, target)

    print(custom.predict(test))
    print(sklearn.predict(test))

    assert np.array_equal(custom.predict(test), sklearn.predict(test))


def test_smarttransformer_notimplemented():

    import pandas as pd

    from ..transformers import SmartTransformer
    from ..transformers import StandardScaler

    df = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    class TestSmartTransformer(SmartTransformer):
        pass

    transformer = TestSmartTransformer()

    with pytest.raises(NotImplementedError) as e:
        transformer.fit(df[["crim"]])

    assert str(e.value) == "WrappedTransformer _get_transformer was not implimented."


def test_smarttransformer_attributeerror():

    import pandas as pd

    from ..transformers import SmartTransformer
    from ..transformers import StandardScaler

    df = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    class TestSmartTransformer(SmartTransformer):
        def _get_transformer(self, X, **fit_params):
            return None

    transformer = TestSmartTransformer()

    with pytest.raises(AttributeError) as e:
        transformer.fit(df[["crim"]])

    assert (
        str(e.value)
        == "Invalid WrappedTransformer. Get transformer returns invalid object"
    )


def test_smarttransformer_invalidtransformer():

    import pandas as pd

    from ..transformers import SmartTransformer
    from ..transformers import StandardScaler

    df = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    class InvalidClass:
        pass

    class TestSmartTransformer(SmartTransformer):
        def _get_transformer(self, X, **fit_params):
            return InvalidClass()

    transformer = TestSmartTransformer()

    with pytest.raises(AttributeError) as e:
        transformer.fit(df[["crim"]])

    assert (
        str(e.value)
        == "Invalid WrappedTransformer. Get transformer returns invalid object"
    )


def test_smarttransformer_function():

    import pandas as pd

    from ..transformers import SmartTransformer
    from ..transformers import StandardScaler

    df = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    class TestSmartTransformer(SmartTransformer):
        def _get_transformer(self, X, **fit_params):
            return StandardScaler()

    smart = TestSmartTransformer()
    smart_data = smart.fit_transform(df[["crim"]])

    std = StandardScaler()
    std_data = std.fit_transform(df[["crim"]])

    assert smart_data.equals(std_data)

    smart.fit(df[["crim"]])
    smart_data = smart.transform(df[["crim"]])

    std.fit(df[["crim"]])
    std_data = std.transform(df[["crim"]])

    assert smart_data.equals(std_data)
