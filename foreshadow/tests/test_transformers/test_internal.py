import pytest


def test_dummy_encoder():
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
    import os

    import pandas as pd
    from foreshadow.transformers.internals import FancyImputer

    heart_path = os.path.join(
        os.path.dirname(__file__), "..", "test_data", "heart-h.csv"
    )
    heart_impute_path = os.path.join(
        os.path.dirname(__file__), "..", "test_data", "heart-h_impute_mean.csv"
    )

    impute = FancyImputer(method="SimpleFill", fill_method="median")
    impute.set_params(**{"fill_method": "mean"})

    df = pd.read_csv(heart_path)

    data = df[["chol"]]

    impute.fit(data)
    out = impute.transform(data)
    truth = pd.read_csv(heart_impute_path, index_col=0)

    assert out.equals(truth)


def test_transformer_fancy_impute_get_params():
    from foreshadow.transformers.internals import FancyImputer

    impute = FancyImputer(method="SimpleFill", fill_method="median")

    assert impute.get_params().get("fill_method", None) == "median"


def test_transformer_fancy_impute_invalid_init():
    from foreshadow.transformers.internals import FancyImputer

    with pytest.raises(ValueError) as e:
        _ = FancyImputer(method="INVALID")

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

    df = pd.DataFrame(
        {"neat": ["apple", "apple", "orange", "apple", "orange"]}
    )
    ohe = OneHotEncoder(use_cat_names=True, handle_unknown="ignore")
    assert ohe.fit(df) == ohe
    assert list(ohe.transform(df)) == [
        "neat_OneHotEncoder_neat_apple",
        "neat_OneHotEncoder_neat_orange",
    ]


def test_transformer_onehotencoder_fit_transform_keep_cols():
    import pandas as pd
    from foreshadow.transformers.externals import OneHotEncoder

    df = pd.DataFrame(
        {"neat": ["apple", "apple", "orange", "apple", "orange"]}
    )
    ohe = OneHotEncoder(
        keep_columns=True,
        name="encoder",
        use_cat_names=True,
        handle_unknown="ignore",
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


def test_prepare_financial():
    import numpy as np
    import pandas as pd
    from foreshadow.transformers.internals import PrepareFinancial

    x = pd.DataFrame(
        [
            "Test",
            "(123)",
            "  123",
            "[123]",
            "123,",
            "123.",
            "-123",
            "123,123",
            "ab123.3",
        ]
    )
    expected = pd.DataFrame(
        [
            np.nan,
            "(123)",
            "123",
            "[123]",
            "123,",
            "123.",
            "-123",
            "123,123",
            "123.3",
        ]
    ).values
    out = PrepareFinancial().fit_transform(x).values

    assert np.all((out == expected) | (pd.isnull(out) == pd.isnull(expected)))


def test_convert_financial_us():
    import numpy as np
    import pandas as pd
    from foreshadow.transformers.internals import ConvertFinancial

    x = pd.DataFrame(
        [
            "0",
            "000",
            "0.9",
            "[0.9]",
            "-.3",
            "30.00",
            "1,000",
            "1.000,000",
            "1.1.1",
        ]
    )
    expected = pd.DataFrame(
        [0.0, 0.0, 0.9, -0.9, -0.3, 30.0, 1000.0, np.nan, np.nan]
    ).values
    out = ConvertFinancial().fit_transform(x).values
    assert np.all((out == expected) | (pd.isnull(out) == pd.isnull(expected)))


def test_convert_financial_eu():
    import numpy as np
    import pandas as pd
    from foreshadow.transformers.internals import ConvertFinancial

    x = pd.DataFrame(
        [
            "0",
            "000",
            "0,9",
            "[0,9]",
            "-,3",
            "30,00",
            "1.000",
            "1,000.000",
            "1.1.1",
        ]
    )
    expected = pd.DataFrame(
        [0, 0, 0.9, -0.9, -0.3, 30.0, 1000.0, np.nan, np.nan]
    ).values
    out = ConvertFinancial(is_euro=True).fit_transform(x).values
    assert np.all((out == expected) | (pd.isnull(out) == pd.isnull(expected)))


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
    assert np.array_equal(
        pd.unique(set_replacement.values.ravel()), np.array([1, 3])
    )


def test_uncommon_remover_strings():
    import numpy as np
    import pandas as pd
    from foreshadow.transformers.internals import UncommonRemover

    x = pd.DataFrame(
        {"A": np.array(["A", "B", "C"] + ["D"] * 400 + ["E"] * 400)}
    )
    standard = UncommonRemover().fit_transform(x)
    set_replacement = UncommonRemover(replacement="D").fit_transform(x)

    assert np.array_equal(
        pd.unique(standard.values.ravel()),
        np.array(["UncommonRemover_Other", "D", "E"], dtype="object"),
    )
    assert np.array_equal(
        pd.unique(set_replacement.values.ravel()), np.array(["D", "E"])
    )


def test_html_remover_basic():
    import numpy as np
    import pandas as pd
    from foreshadow.transformers.internals import HTMLRemover

    df = pd.DataFrame(
        ["<h1>Header<h1/>", "Normal Text", "<br/><br/>More text"]
    )
    df_assert = pd.DataFrame(["Header", "Normal Text", "More text"])

    hr = HTMLRemover()

    assert np.array_equal(
        hr.fit_transform(df).values.ravel(), df_assert.values.ravel()
    )


def test_html_remover_is_html():
    from foreshadow.transformers.internals import HTMLRemover

    html = "<b>Real Tag</b> Test"
    not_html = "<not tag>"

    assert HTMLRemover.is_html(html)
    assert not HTMLRemover.is_html(not_html)


def test_to_string_tf():
    import numpy as np
    import pandas as pd
    from foreshadow.transformers.internals import ToString

    data = [0, 1, 2, 3, np.nan]
    arr = np.array(data)
    df = pd.DataFrame(data)

    expected = ["0.0", "1.0", "2.0", "3.0", "nan"]

    ts = ToString()

    assert expected == ts.transform(arr).values.ravel().tolist()
    assert expected == ts.transform(df).values.ravel().tolist()


def test_tfidf_and_sparse_processing_core():
    import numpy as np
    import pandas as pd
    from foreshadow.transformers.internals import FixedTfidfVectorizer

    X1 = ["Test", "Another test", "How about another"]
    X2 = pd.DataFrame(["Test", "Another test", "How about another"])

    exp = [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.7071067811865476, 0.0, 0.7071067811865476],
        [0.6227660078332259, 0.4736296010332684, 0.6227660078332259, 0.0],
    ]
    exp2 = [["test"], ["another", "test"], ["about", "another", "how"]]

    tfidf = FixedTfidfVectorizer()

    rslt1 = tfidf.fit_transform(X1)
    rslt2 = tfidf.fit(X2).transform(X2)

    assert np.allclose(rslt1, exp)
    assert np.allclose(rslt2, exp)
    assert tfidf.inverse_transform(rslt1) == exp2
