import pytest


def test_box_cox():
    import numpy as np
    import pandas as pd
    import scipy.stats as ss

    from foreshadow.transformers import BoxCoxTransformer

    np.random.seed(0)
    data = pd.DataFrame(ss.lognorm.rvs(size=100, s=0.954))
    bc = BoxCoxTransformer()
    bc_data = bc.fit_transform(data)
    assert ss.shapiro(bc_data)[1] > 0.05
    assert np.allclose(
        data.values.ravel(), bc.inverse_transform(bc_data).values.ravel()
    )


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
