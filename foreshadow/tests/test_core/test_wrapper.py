"""Test the wrapper.py functionality."""
import pytest

from foreshadow.utils.testing import get_file_path


def test_transformer_wrapper_init():
    from foreshadow.concrete import StandardScaler

    scaler = StandardScaler(name="test-scaler", keep_columns=True)

    assert scaler.name == "test-scaler"
    assert scaler.keep_columns is True


def test_transformer_wrapper_no_init():
    from foreshadow.base import BaseEstimator, TransformerMixin
    from foreshadow.wrapper import pandas_wrap

    class NewTransformer(BaseEstimator, TransformerMixin):
        pass

    trans = pandas_wrap(NewTransformer)
    _ = trans()

    assert hasattr(trans.__init__, "__defaults__")


def test_transformer_wrapper_function():
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler as StandardScaler
    from foreshadow.concrete import StandardScaler as CustomScaler

    boston_path = get_file_path("data", "boston_housing.csv")

    df = pd.read_csv(boston_path)

    custom = CustomScaler()
    sklearn = StandardScaler()

    custom.fit(df[["crim"]])
    sklearn.fit(df[["crim"]])

    custom_tf = custom.transform(df[["crim"]])
    sklearn_tf = sklearn.transform(df[["crim"]])

    assert np.array_equal(custom_tf.values, sklearn_tf)

    custom_tf = custom.fit_transform(df[["crim"]])
    sklearn_tf = sklearn.fit_transform(df[["crim"]])

    assert np.array_equal(custom_tf.values, sklearn_tf)


def test_transformer_wrapper_empty_input():
    import numpy as np
    import pandas as pd

    from sklearn.preprocessing import StandardScaler as StandardScaler
    from foreshadow.concrete import StandardScaler as CustomScaler

    df = pd.DataFrame({"A": np.array([])})

    with pytest.raises(ValueError):
        StandardScaler().fit(df)
    with pytest.raises(ValueError):
        CustomScaler().fit(df)
