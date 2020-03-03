import pytest

from foreshadow.utils.testing import get_file_path


def test_nan_filler():
    import pandas as pd
    import numpy as np

    from foreshadow.concrete import NaNFiller
    from foreshadow.utils import Constant

    data = pd.DataFrame(
        {
            "a": ["123", "a", "b", np.nan],
            "b": [np.nan, "q", "w", "v"],
            "c": [np.nan, "1", "0", "1"],
        }
    )

    check = pd.DataFrame(
        {
            "a": ["123", "a", "b", Constant.NAN_FILL_VALUE],
            "b": [Constant.NAN_FILL_VALUE, "q", "w", "v"],
            "c": [Constant.NAN_FILL_VALUE, "1", "0", "1"],
        }
    )

    filler = NaNFiller()
    df_transformed = filler.transform(data)
    assert check.equals(df_transformed)

    inverse_transformed = filler.inverse_transform(df_transformed)
    assert data.equals(inverse_transformed)


def test_dummy_encoder():
    import pandas as pd

    from foreshadow.concrete import DummyEncoder

    data = pd.DataFrame({"test": ["a", "a,b,c", "a,b", "a,c"]})
    de = DummyEncoder()
    de.fit(data)
    df = de.transform(data)

    check = pd.DataFrame(
        {"a": [1, 1, 1, 1], "b": [0, 1, 1, 0], "c": [0, 1, 0, 1]}
    )

    assert check.equals(df)


@pytest.mark.parametrize("deep", [True, False])
def test_dummy_encoder_get_params_keys(deep):
    """Test that the desired keys show up for the DummyEncoder object.

    Args:
        deep: deep param to get_params

    """
    from foreshadow.concrete import DummyEncoder

    de = DummyEncoder()
    params = de.get_params(deep=deep)

    desired_keys = ["delimeter", "other_cutoff", "other_name"]
    for key in desired_keys:
        assert key in params


def test_dummy_encoder_other():
    import pandas as pd

    from foreshadow.concrete import DummyEncoder

    data = pd.DataFrame(
        {"test": ["a", "a,b,c", "a,b", "a,c,d", "a,b,c", "a,b,c", "a,b,c,e"]}
    )
    de = DummyEncoder(other_cutoff=0.25)
    de.fit(data)
    df = de.transform(data)
    check = pd.DataFrame(
        {
            "a": [1, 1, 1, 1, 1, 1, 1],
            "b": [0, 1, 1, 0, 1, 1, 1],
            "c": [0, 1, 0, 1, 1, 1, 1],
            "other": [0, 0, 0, 1, 0, 0, 1],
        }
    )

    assert check.equals(df)


@pytest.mark.parametrize("deep", [True, False])
def test_label_encoder_get_params_keys(deep):
    """Test that the desired keys show up for the LabelEncoder object.

        Args:
            deep: deep param to get_params

        """
    from foreshadow.concrete import FixedLabelEncoder

    fle = FixedLabelEncoder()
    params = fle.get_params(deep=deep)

    desired_keys = ["encoder"]
    for key in desired_keys:
        assert key in params


def test_transformer_fancy_impute_set_params():
    import numpy as np
    import pandas as pd
    from foreshadow.concrete import FancyImputer

    impute_kwargs = {"fill_method": "mean"}

    impute = FancyImputer(method="SimpleFill", impute_kwargs=impute_kwargs)
    heart_path = get_file_path("data", "heart-h.csv")
    heart_impute_path = get_file_path("data", "heart-h_impute_mean.csv")

    df = pd.read_csv(heart_path)

    data = df[["chol"]]

    impute.fit(data)
    out = impute.transform(data)
    truth = pd.read_csv(heart_impute_path, index_col=0)

    assert np.array_equal(out, truth)


def test_transformer_fancy_impute_get_params():
    from foreshadow.concrete import FancyImputer

    impute_kwargs = {"fill_method": "median"}
    impute = FancyImputer(method="SimpleFill", impute_kwargs=impute_kwargs)

    impute_kwargs = impute.get_params().get("impute_kwargs", None)
    assert impute_kwargs is not None
    assert impute_kwargs.get("fill_method", None) == "median"


def test_transformer_fancy_impute_invalid_init():
    from foreshadow.concrete import FancyImputer

    with pytest.raises(ValueError) as e:
        _ = FancyImputer(method="INVALID")

    assert str(e.value) == (
        "Invalid method. Possible values are BiScaler, KNN, "
        "NuclearNormMinimization and SoftImpute"
    )


def test_transformer_fancy_impute_invalid_params():
    from foreshadow.concrete import FancyImputer

    with pytest.raises(ValueError) as e:
        impute = FancyImputer(
            method="SimpleFill", impute_kwargs={"fill_method": "mean"}
        )
        impute.set_params(**{"method": "INVALID"})

    assert str(e.value) == (
        "Invalid method. Possible values are BiScaler, KNN, "
        "NuclearNormMinimization and SoftImpute"
    )


def test_transformer_onehotencoder_fit_transform():
    import pandas as pd
    from foreshadow.concrete import OneHotEncoder

    df = pd.DataFrame(
        {"neat": ["apple", "apple", "orange", "apple", "orange"]}
    )
    ohe = OneHotEncoder(use_cat_names=True, handle_unknown="ignore")
    assert ohe.fit(df) == ohe
    assert list(ohe.transform(df)) == ["neat_apple", "neat_orange"]


def test_transformer_onehotencoder_fit_transform_keep_cols():
    import pandas as pd
    from foreshadow.concrete import OneHotEncoder

    df = pd.DataFrame(
        {"neat": ["apple", "apple", "orange", "apple", "orange"]}
    )
    ohe = OneHotEncoder(
        use_cat_names=True,
        handle_unknown="ignore",
        name="encoder",
        keep_columns=True,
    )
    assert ohe.fit(df) == ohe
    assert list(ohe.transform(df)) == ["neat", "neat_apple", "neat_orange"]


def test_drop_transformer_above_thresh():
    import numpy as np
    import pandas as pd
    from foreshadow.concrete import DropFeature

    x = pd.DataFrame({"A": np.arange(10)})

    assert np.array_equal(
        x.values.ravel(), DropFeature().fit_transform(x).values.ravel()
    )


def test_drop_transformer_below_thresh():
    import numpy as np
    import pandas as pd
    from foreshadow.concrete import DropFeature

    # default thresh is 0.3
    x = pd.DataFrame({"A": np.array([np.nan] * 8 + [0.1, 0.1])})

    assert DropFeature().fit_transform(x).values.size == 0


def test_drop_transformer_disallow_inverse():
    import numpy as np
    import pandas as pd
    from foreshadow.concrete import DropFeature

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
    from foreshadow.concrete import DropFeature

    x = pd.DataFrame({"A": np.array([np.nan] * 8 + [0.1, 0.1])})
    drop_f = DropFeature().fit(x)
    inv = drop_f.inverse_transform([1, 2, 10])

    assert np.array_equal(inv.values.ravel(), np.array([]))


def test_drop_transformer_string_input():
    import uuid
    import numpy as np
    import pandas as pd
    from foreshadow.concrete import DropFeature

    x = pd.DataFrame({"A": np.array([str(uuid.uuid4()) for _ in range(40)])})
    assert np.array_equal(
        x.values.ravel(), DropFeature().fit_transform(x).values.ravel()
    )


def test_prepare_financial():
    import numpy as np
    import pandas as pd
    from foreshadow.concrete import PrepareFinancial

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
    from foreshadow.concrete import ConvertFinancial

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
    from foreshadow.concrete import ConvertFinancial

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
    from foreshadow.concrete import UncommonRemover

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
    from foreshadow.concrete import UncommonRemover

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
    from foreshadow.concrete import HTMLRemover

    df = pd.DataFrame(
        ["<h1>Header<h1/>", "Normal Text", "<br/><br/>More text"]
    )
    df_assert = pd.DataFrame(["Header", "Normal Text", "More text"])

    hr = HTMLRemover()

    assert np.array_equal(
        hr.fit_transform(df).values.ravel(), df_assert.values.ravel()
    )


def test_html_remover_is_html():
    from foreshadow.concrete import HTMLRemover

    html = "<b>Real Tag</b> Test"
    not_html = "<not tag>"

    assert HTMLRemover.is_html(html)
    assert not HTMLRemover.is_html(not_html)


def test_to_string_tf():
    import numpy as np
    import pandas as pd
    from foreshadow.concrete import ToString

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
    from foreshadow.concrete import FixedTfidfVectorizer

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
    print(rslt1)
    assert np.allclose(rslt1, exp)
    assert np.allclose(rslt2, exp)
    rslt1_inverse = tfidf.inverse_transform(rslt1).values  # TODO move to
    # TODO non DataFrame output
    rslt1_inverse = [x[0] for x in rslt1_inverse]
    assert np.array_equal(rslt1_inverse, exp2)
