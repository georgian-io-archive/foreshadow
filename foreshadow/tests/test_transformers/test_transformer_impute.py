import pytest


def test_transformer_impute_simple_none():

    import pandas as pd
    from foreshadow.transformers import SimpleImputer

    impute = SimpleImputer(threshold=0.05)
    df = pd.read_csv("./foreshadow/tests/data/heart-h.csv")

    data = df[["chol"]]

    impute.fit(data)
    out = impute.transform(data)

    assert data.equals(out)


def test_transformer_impute_simple_mean():

    import pandas as pd
    from foreshadow.transformers import SimpleImputer

    impute = SimpleImputer()
    df = pd.read_csv("./foreshadow/tests/data/heart-h.csv")

    data = df[["chol"]]

    impute.fit(data)
    out = impute.transform(data)
    truth = pd.read_csv("./foreshadow/tests/data/heart-h_impute_mean.csv", index_col=0)

    assert out.equals(truth)


def test_transformer_impute_simple_median():

    import pandas as pd
    import numpy as np
    from foreshadow.transformers import SimpleImputer

    impute = SimpleImputer()
    df = pd.read_csv("./foreshadow/tests/data/heart-h.csv")

    data = df["chol"].values
    data = np.append(data, [2 ** 10] * 100)

    impute.fit(data)
    out = impute.transform(data)
    truth = pd.read_csv(
        "./foreshadow/tests/data/heart-h_impute_median.csv", index_col=0
    )

    assert out.equals(truth)


def test_transformer_impute_multiple():

    import numpy as np
    import pandas as pd
    from foreshadow.transformers import MultiImputer

    impute = MultiImputer()
    df = pd.read_csv("./foreshadow/tests/data/heart-h.csv")

    data = df[["thalach", "chol", "trestbps", "age"]]

    impute.fit(data)
    out = impute.transform(data)
    truth = pd.read_csv("./foreshadow/tests/data/heart-h_impute_multi.csv", index_col=0)

    assert np.allclose(truth.values, out.values)


def test_transformer_fancy_impute_set_params():

    import pandas as pd
    from foreshadow.transformers import FancyImputer

    impute = FancyImputer(method="SimpleFill", fill_method="median")
    impute.set_params(**{"fill_method": "mean"})

    df = pd.read_csv("./foreshadow/tests/data/heart-h.csv")

    data = df[["chol"]]

    impute.fit(data)
    out = impute.transform(data)
    truth = pd.read_csv("./foreshadow/tests/data/heart-h_impute_mean.csv", index_col=0)

    assert out.equals(truth)


def test_transformer_fancy_impute_get_params():

    from foreshadow.transformers import FancyImputer

    impute = FancyImputer(method="SimpleFill", fill_method="median")

    assert impute.get_params().get("fill_method", None) == "median"


def test_transformer_fancy_impute_invalid_init():

    from foreshadow.transformers import FancyImputer

    with pytest.raises(ValueError) as e:
        impute = FancyImputer(method="INVALID")

    assert str(e.value) == (
        "Invalid method. Possible values are BiScaler, KNN, "
        "NuclearNormMinimization and SoftImpute"
    )


def test_transformer_fancy_impute_invalid_params():

    from foreshadow.transformers import FancyImputer

    with pytest.raises(ValueError) as e:
        impute = FancyImputer(method="SimpleFill", fill_method="mean")
        impute.set_params(**{"method": "INVALID"})

    assert str(e.value) == (
        "Invalid method. Possible values are BiScaler, KNN, "
        "NuclearNormMinimization and SoftImpute"
    )
