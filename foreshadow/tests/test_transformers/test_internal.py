import pytest


def test_dummy_encoder():
    import numpy as np
    import pandas as pd

    from foreshadow.transformers.internals import DummyEncoder

    data = pd.DataFrame({"test": ["a", "a,b,c", "a,b", "a,c"]})
    de = DummyEncoder()
    de.fit(data)
    df = de.transform(data)

    check = pd.DataFrame(
        {
            "test_DummyEncoder_a": [1, 1, 1, 1],
            "test_DummyEncoder_b": [0, 1, 1, 0],
            "test_DummyEncoder_c": [0, 1, 0, 1],
        }
    )

    assert check.equals(df)


def test_dummy_encoder_other():
    import numpy as np
    import pandas as pd

    from foreshadow.transformers.internals import DummyEncoder

    data = pd.DataFrame(
        {"test": ["a", "a,b,c", "a,b", "a,c,d", "a,b,c", "a,b,c", "a,b,c,e"]}
    )
    de = DummyEncoder(other_cutoff=0.25)
    de.fit(data)
    df = de.transform(data)

    check = pd.DataFrame(
        {
            "test_DummyEncoder_a": [1, 1, 1, 1, 1, 1, 1],
            "test_DummyEncoder_b": [0, 1, 1, 0, 1, 1, 1],
            "test_DummyEncoder_c": [0, 1, 0, 1, 1, 1, 1],
            "test_DummyEncoder_other": [0, 0, 0, 1, 0, 0, 1],
        }
    )

    assert check.equals(df)


def test_box_cox():
    import numpy as np
    import pandas as pd
    import scipy.stats as ss

    from foreshadow.transformers.internals import BoxCox

    np.random.seed(0)
    data = pd.DataFrame(ss.lognorm.rvs(size=100, s=0.954))
    bc = BoxCox()
    bc_data = bc.fit_transform(data)
    assert ss.shapiro(bc_data)[1] > 0.05
    assert np.allclose(
        data.values.ravel(), bc.inverse_transform(bc_data).values.ravel()
    )


def test_transformer_fancy_impute_set_params():
    import pandas as pd
    from foreshadow.transformers.internals import FancyImputer

    impute = FancyImputer(method="SimpleFill", fill_method="median")
    impute.set_params(**{"fill_method": "mean"})

    df = pd.read_csv("./foreshadow/tests/test_data/heart-h.csv")

    data = df[["chol"]]

    impute.fit(data)
    out = impute.transform(data)
    truth = pd.read_csv(
        "./foreshadow/tests/test_data/heart-h_impute_mean.csv", index_col=0
    )

    assert out.equals(truth)


def test_transformer_fancy_impute_get_params():
    from foreshadow.transformers.internals import FancyImputer

    impute = FancyImputer(method="SimpleFill", fill_method="median")

    assert impute.get_params().get("fill_method", None) == "median"


def test_transformer_fancy_impute_invalid_init():
    from foreshadow.transformers.internals import FancyImputer

    with pytest.raises(ValueError) as e:
        impute = FancyImputer(method="INVALID")

    assert str(e.value) == (
        "Invalid method. Possible values are BiScaler, KNN, "
        "NuclearNormMinimization and SoftImpute"
    )


def test_transformer_fancy_impute_invalid_params():
    from foreshadow.transformers.internals import FancyImputer

    with pytest.raises(ValueError) as e:
        impute = FancyImputer(method="SimpleFill", fill_method="mean")
        impute.set_params(**{"method": "INVALID"})

    assert str(e.value) == (
        "Invalid method. Possible values are BiScaler, KNN, "
        "NuclearNormMinimization and SoftImpute"
    )


def test_transformer_onehotencoder_fit_transform():
    import pandas as pd
    from foreshadow.transformers.externals import OneHotEncoder

    df = pd.DataFrame({"neat": ["apple", "apple", "orange", "apple", "orange"]})
    ohe = OneHotEncoder(use_cat_names=True, handle_unknown="ignore")
    assert ohe.fit(df) == ohe
    assert list(ohe.transform(df)) == [
        "neat_OneHotEncoder_neat_apple",
        "neat_OneHotEncoder_neat_orange",
    ]


def test_transformer_onehotencoder_fit_transform_keep_cols():
    import pandas as pd
    from foreshadow.transformers.externals import OneHotEncoder

    df = pd.DataFrame({"neat": ["apple", "apple", "orange", "apple", "orange"]})
    ohe = OneHotEncoder(
        keep_columns=True, name="encoder", use_cat_names=True, handle_unknown="ignore"
    )
    assert ohe.fit(df) == ohe
    assert list(ohe.transform(df)) == [
        "neat_encoder_origin_0",
        "neat_encoder_neat_apple",
        "neat_encoder_neat_orange",
    ]


def test_drop_transformer_above_thresh():
    import numpy as np
    import pandas as pd
    from foreshadow.transformers.internals import DropFeature

    x = pd.DataFrame({"A": np.arange(10)})

    assert np.array_equal(
        x.values.ravel(), DropFeature().fit_transform(x).values.ravel()
    )


def test_drop_transformer_below_thresh():
    import numpy as np
    import pandas as pd
    from foreshadow.transformers.internals import DropFeature

    # default thresh is 0.3
    x = pd.DataFrame({"A": np.array([np.nan] * 8 + [0.1, 0.1])})

    assert DropFeature().fit_transform(x).values.size == 0


def test_drop_transformer_disallow_inverse():
    import numpy as np
    import pandas as pd
    from foreshadow.transformers.internals import DropFeature

    x = pd.DataFrame({"A": np.array([np.nan] * 8 + [0.1, 0.1])})
    drop_f = DropFeature(raise_on_inverse=True).fit(x)

    with pytest.raises(ValueError) as e:
        drop_f.inverse_transform([1, 2, 10])

    assert str(e.value) == (
        "inverse_transform is not permitted on this DropFeature instance"
    )


def test_drop_transformer_default_inverse():
    import numpy as np
    import pandas as pd
    from foreshadow.transformers.internals import DropFeature

    x = pd.DataFrame({"A": np.array([np.nan] * 8 + [0.1, 0.1])})
    drop_f = DropFeature().fit(x)
    inv = drop_f.inverse_transform([1, 2, 10])

    assert np.array_equal(inv.values.ravel(), np.array([]))


def test_drop_transformer_string_input():
    import uuid
    import numpy as np
    import pandas as pd
    from foreshadow.transformers.internals import DropFeature

    x = pd.DataFrame({"A": np.array([str(uuid.uuid4()) for _ in range(40)])})
    assert np.array_equal(
        x.values.ravel(), DropFeature().fit_transform(x).values.ravel()
    )


def test_uncommon_remover_integers():
    import numpy as np
    import pandas as pd
    from foreshadow.transformers.internals import UncommonRemover

    x = pd.DataFrame({"A": np.array([0, 2, 10] + [1] * 400 + [3] * 400)})
    standard = UncommonRemover().fit_transform(x)
    set_replacement = UncommonRemover(replacement=1).fit_transform(x)

    assert np.array_equal(
        pd.unique(standard.values.ravel()),
        np.array(["UncommonRemover_Other", 1, 3], dtype="object"),
    )
    assert np.array_equal(pd.unique(set_replacement.values.ravel()), np.array([1, 3]))


def test_uncommon_remover_strings():
    import numpy as np
    import pandas as pd
    from foreshadow.transformers.internals import UncommonRemover

    x = pd.DataFrame({"A": np.array(["A", "B", "C"] + ["D"] * 400 + ["E"] * 400)})
    standard = UncommonRemover().fit_transform(x)
    set_replacement = UncommonRemover(replacement="D").fit_transform(x)

    assert np.array_equal(
        pd.unique(standard.values.ravel()),
        np.array(["UncommonRemover_Other", "D", "E"], dtype="object"),
    )
    assert np.array_equal(
        pd.unique(set_replacement.values.ravel()), np.array(["D", "E"])
    )
