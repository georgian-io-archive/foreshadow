import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from foreshadow.cachemanager import CacheManager
from foreshadow.intents import IntentType
from foreshadow.utils import AcceptedKey, TruncatedSVDWrapper
from foreshadow.utils.testing import get_file_path


@pytest.fixture()
def smart_child():
    """Get a defined SmartTransformer subclass, TestSmartTransformer.

    Note:
        Always returns StandardScaler.

    """
    from foreshadow.smart import SmartTransformer
    from foreshadow.concrete import StandardScaler

    class TestSmartTransformer(SmartTransformer):
        def pick_transformer(self, X, y=None, **fit_params):
            return StandardScaler()

    yield TestSmartTransformer


@pytest.fixture()
def smart_params():
    """Get the params for a defined SmartTransformer subclass."""
    yield {
        "check_wrapped": True,
        "cache_manager": None,
        "force_reresolve": False,
        "keep_columns": False,
        "name": None,
        "should_resolve": True,
        "transformer": None,
        "y_var": False,
    }


@pytest.mark.parametrize("deep", [True, False])
def test_smart_get_params_default(smart_child, smart_params, deep):
    """Ensure that default get_params works.

    Args:
        smart_child: a smart instance
        deep: deep param to get_params

    """
    smart = smart_child()
    params = smart.get_params(deep=deep)
    assert smart_params == params


@pytest.mark.parametrize("initial_transformer", [None, "StandardScaler"])
def test_smart_get_params_deep(smart_child, smart_params, initial_transformer):
    """Test that smart.get_params(deep=True) functions as desired.

    Args:
        smart_child: SmartTransformer subclass instance fixture
        smart_params: default params for above
        initial_transformer: the transformer to set on smart.transformer for
            the test.

    """
    smart = smart_child()
    smart.transformer = initial_transformer
    try:
        nested_params = smart.transformer.get_params(deep=True)
        nested_params = {
            "transformer__" + key: val for key, val in nested_params.items()
        }
        nested_params["should_resolve"] = False
    except AttributeError:  # case of None
        nested_params = {}
    nested_params["transformer"] = smart.transformer
    smart_params.update(nested_params)
    assert smart.get_params(True) == smart_params


@pytest.mark.parametrize("initial_transformer", [None, "StandardScaler"])
def test_smart_set_params_default(smart_child, initial_transformer):
    """Test setting both transformer and its parameters simultaneously works.

    Current sklearn implementation does not allow this and we created our
    own BaseEstimator to allow this functionality.

    Args:
        smart_child: smart instance
        initial_transformer: the initial transformer to put before trying to
        set_params().

    """
    from foreshadow.concrete import StandardScaler

    smart = smart_child()
    smart.transformer = initial_transformer
    params = {"transformer": "StandardScaler", "transformer__with_std": False}
    smart.set_params(**params)
    check = {
        "check_wrapped": True,
        "cache_manager": None,
        "force_reresolve": False,
        "keep_columns": False,
        "name": None,
        "should_resolve": False,
        "y_var": False,
        "transformer__with_std": False,
        "transformer__copy": True,
        "transformer__with_mean": True,
    }
    params = smart.get_params()
    assert isinstance(params.pop("transformer"), StandardScaler)
    assert check == params


def test_smart_emtpy_input():
    import numpy as np

    from foreshadow.smart import Scaler

    normal_data = np.array([])
    smart_scaler = Scaler()

    with pytest.raises(ValueError):
        smart_scaler.fit_transform(normal_data).values.size == 0


def test_smart_scaler_normal():
    import numpy as np
    import scipy.stats as ss

    from foreshadow.smart import Scaler
    from foreshadow.concrete import StandardScaler

    np.random.seed(0)
    normal_data = ss.norm.rvs(size=100)
    smart_scaler = Scaler()
    assert isinstance(
        smart_scaler.fit(normal_data).transformer, StandardScaler
    )


def test_smart_scaler_unifrom():
    import numpy as np
    import scipy.stats as ss

    from foreshadow.smart import Scaler
    from foreshadow.concrete import MinMaxScaler

    np.random.seed(0)
    uniform_data = ss.uniform.rvs(size=100)
    smart_scaler = Scaler()
    assert isinstance(smart_scaler.fit(uniform_data).transformer, MinMaxScaler)


def test_smart_scaler_neither():
    import numpy as np
    import scipy.stats as ss

    from foreshadow.smart import Scaler
    from sklearn.pipeline import Pipeline

    np.random.seed(0)
    lognorm_data = ss.lognorm.rvs(size=100, s=0.954)  # one example
    smart_scaler = Scaler()
    assert isinstance(smart_scaler.fit(lognorm_data).transformer, Pipeline)


def test_smart_encoder_less_than_30_levels():
    import numpy as np

    from foreshadow.smart import CategoricalEncoder
    from foreshadow.concrete import OneHotEncoder
    from foreshadow.concrete import NaNFiller

    np.random.seed(0)
    leq_30_random_data = np.random.choice(30, size=500)
    smart_coder = CategoricalEncoder()
    transformer = smart_coder.fit(leq_30_random_data).transformer
    assert isinstance(transformer, Pipeline)
    assert isinstance(transformer.steps[0][1], NaNFiller)
    assert isinstance(transformer.steps[1][1], OneHotEncoder)
    assert len(transformer.steps) == 2


def test_smart_encoder_more_than_30_levels():
    import numpy as np

    from foreshadow.smart import CategoricalEncoder
    from foreshadow.concrete import HashingEncoder
    from foreshadow.concrete import NaNFiller

    np.random.seed(0)
    gt_30_random_data = np.random.choice(31, size=500)
    gt_30_random_data = [item for item in gt_30_random_data.astype(str)]
    gt_30_random_data[0] = np.nan
    smart_coder = CategoricalEncoder()
    transformer = smart_coder.fit(gt_30_random_data).transformer
    assert isinstance(transformer, Pipeline)
    assert isinstance(transformer.steps[0][1], NaNFiller)
    assert isinstance(transformer.steps[1][1], HashingEncoder)
    assert len(transformer.steps) == 2
    res = transformer.transform(gt_30_random_data)
    assert len(res.columns) == 30


def test_smart_encoder_more_than_30_levels_that_reduces():
    import numpy as np

    from foreshadow.smart import CategoricalEncoder
    from foreshadow.concrete import OneHotEncoder

    np.random.seed(0)
    gt_30_random_data = np.concatenate(
        [np.random.choice(29, size=500), np.array([31, 32, 33, 34, 35, 36])]
    )
    smart_coder = CategoricalEncoder()
    assert isinstance(
        smart_coder.fit(gt_30_random_data).transformer.steps[-1][1],
        OneHotEncoder,
    )


def test_smart_encoder_y_var():
    import numpy as np
    import pandas as pd

    from foreshadow.smart import CategoricalEncoder
    from foreshadow.concrete import FixedLabelEncoder as LabelEncoder

    y_df = pd.DataFrame({"A": np.array([1, 2, 10] * 3)})
    smart_coder = CategoricalEncoder(y_var=True)

    assert isinstance(smart_coder.fit(y_df).transformer, LabelEncoder)
    assert np.array_equal(
        smart_coder.transform(y_df).values.ravel(), np.array([0, 1, 2] * 3)
    )


@pytest.mark.skip("Need to fix!")
def test_smart_impute_simple_none():
    import numpy as np
    import pandas as pd
    from foreshadow.smart import SimpleFillImputer

    heart_path = get_file_path("data", "heart-h.csv")

    impute = SimpleFillImputer(threshold=0.05)
    df = pd.read_csv(heart_path)

    data = df[["chol"]]

    impute.fit(data)
    out = impute.transform(data)

    assert np.allclose(data, out, equal_nan=True)


def test_smart_impute_simple_mean():
    import numpy as np
    import pandas as pd
    from foreshadow.smart import SimpleFillImputer

    heart_path = get_file_path("data", "heart-h.csv")
    heart_impute_path = get_file_path("data", "heart-h_impute_mean.csv")

    impute = SimpleFillImputer()
    df = pd.read_csv(heart_path)

    data = df[["chol"]]

    impute.fit(data)
    out = impute.transform(data)
    truth = pd.read_csv(heart_impute_path, index_col=0)

    assert np.array_equal(out, truth)


def test_smart_impute_simple_median():
    import pandas as pd
    import numpy as np
    from foreshadow.smart import SimpleFillImputer

    heart_path = get_file_path("data", "heart-h.csv")
    heart_impute_path = get_file_path("data", "heart-h_impute_median.csv")

    impute = SimpleFillImputer()
    df = pd.read_csv(heart_path)

    data = df["chol"].values
    data = np.append(data, [2 ** 10] * 100)

    impute.fit(data)
    out = impute.transform(data)
    truth = pd.read_csv(heart_impute_path, index_col=0)

    assert np.array_equal(out, truth)


def test_smart_impute_multiple():
    import numpy as np
    import pandas as pd
    from foreshadow.smart import MultiImputer

    heart_path = get_file_path("data", "heart-h.csv")
    heart_impute_path = get_file_path("data", "heart-h_impute_multi.csv")

    impute = MultiImputer()
    df = pd.read_csv(heart_path)

    data = df[["thalach", "chol", "trestbps", "age"]]

    impute.fit(data)
    out = impute.transform(data)
    truth = pd.read_csv(heart_impute_path, index_col=0)

    assert np.allclose(truth.values, out.values)


def test_smart_impute_multiple_none():
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from foreshadow.smart import MultiImputer
    from foreshadow.utils import PipelineStep

    boston_path = get_file_path("data", "boston_housing.csv")

    impute = MultiImputer()
    df = pd.read_csv(boston_path)

    data = df[["crim", "nox", "indus"]]

    impute.fit(data)
    impute.transform(data)

    assert isinstance(impute.transformer, Pipeline)
    assert impute.transformer.steps[0][PipelineStep["NAME"]] == "null"


@pytest.mark.skip("THIS IS IMPORTANT FIX")
def test_preprocessor_hashencoder_no_name_collision():
    # This test is expected to only do up to DataCleaning right now.
    import uuid
    import numpy as np
    import pandas as pd
    from foreshadow.preparer import DataPreparer
    from foreshadow.cachemanager import CacheManager

    cat1 = [str(uuid.uuid4()) for _ in range(40)]
    cat2 = [str(uuid.uuid4()) for _ in range(40)]

    input = pd.DataFrame(
        {
            "col1": np.random.choice(cat1, 1000),
            "col2": np.random.choice(cat2, 1000),
        }
    )

    dp = DataPreparer(cache_manager=CacheManager())
    output = dp.fit_transform(input)
    # since the number of categories for each column are above 30,
    # HashingEncoder will be used with 30 components. The transformed output
    # should have in total 60 unique names.
    assert len(set(output.columns)) == 60


@pytest.mark.skip("Turning off the dummyencoder feature temporarily")
def test_smart_encoder_delimmited():
    import pandas as pd
    from foreshadow.smart import CategoricalEncoder
    from foreshadow.concrete import DummyEncoder
    from foreshadow.concrete import NaNFiller

    data = pd.DataFrame({"test": ["a", "a,b,c", "a,b", "a,c"]})
    smart_coder = CategoricalEncoder()
    transformer = smart_coder.fit(data).transformer

    assert isinstance(transformer, Pipeline)
    assert isinstance(transformer.steps[0][1], NaNFiller)
    assert isinstance(transformer.steps[1][1], DummyEncoder)
    assert len(transformer.steps) == 2


def test_smart_encoder_more_than_30_levels_with_overwritten_cutoff():
    import numpy as np
    from foreshadow.smart import CategoricalEncoder
    from foreshadow.concrete import OneHotEncoder
    from foreshadow.concrete import NaNFiller

    np.random.seed(0)
    gt_30_random_data = np.random.choice(31, size=500)
    smart_coder = CategoricalEncoder(unique_num_cutoff=35)
    transformer = smart_coder.fit(gt_30_random_data).transformer

    assert isinstance(transformer, Pipeline)
    assert isinstance(transformer.steps[0][1], NaNFiller)
    assert isinstance(transformer.steps[1][1], OneHotEncoder)
    assert len(transformer.steps) == 2


def test_smart_financial_cleaner_us():
    import numpy as np
    import pandas as pd
    from foreshadow.smart import FinancialCleaner

    x = pd.DataFrame(
        [
            "Test",
            "0",
            "000",
            "1,000",
            "0.9",
            "[0.9]",
            "-.3",
            "30.00",
            "3,000.35",
        ]
    )
    expected = pd.DataFrame(
        [np.nan, 0.0, 0.0, 1000, 0.9, -0.9, -0.3, 30.0, 3000.35]
    ).values
    out = FinancialCleaner().fit_transform(x).values

    assert np.all((out == expected) | (pd.isnull(out) == pd.isnull(expected)))


def test_smart_financial_cleaner_eu():
    import numpy as np
    import pandas as pd
    from foreshadow.smart import FinancialCleaner

    x = pd.DataFrame(
        [
            "Test",
            "0",
            "000",
            "1.000",
            "0,9",
            "[0,9]",
            "-,3",
            "30,00",
            "3.000,35",
        ]
    )
    expected = pd.DataFrame(
        [np.nan, 0.0, 0.0, 1000, 0.9, -0.9, -0.3, 30.0, 3000.35]
    ).values
    out = FinancialCleaner().fit_transform(x).values

    assert np.all((out == expected) | (pd.isnull(out) == pd.isnull(expected)))


def test_smart_text():
    import pandas as pd

    from foreshadow.smart import TextEncoder

    X1 = pd.DataFrame(
        data={
            "col1": ["abc", "def", "1321", "tester"],
            "col2": ["okay", "This is a test", "whatup", "gg"],
        }
    )

    manager = CacheManager()
    manager[AcceptedKey.INTENT, "col1"] = IntentType.TEXT

    encoder1 = TextEncoder(n_components=3)
    tf1 = encoder1.fit(X1)
    X1_transformed = tf1.transform(X=X1)
    print()
    print(X1_transformed)

    assert isinstance(tf1.transformer.steps[-2][1], TfidfVectorizer)
    assert isinstance(tf1.transformer.steps[-1][1], TruncatedSVDWrapper)

    # X2 = pd.DataFrame(
    #     data=["<p> Hello </p>", "World", "<h1> Tag </h1>", 123],
    #     columns=["col2"],
    # )
    #
    # encoder2 = TextEncoder()
    # tf2 = encoder2.fit(X2)
    # X2_transformed = tf2.transform(X=X2)
    # print(X2_transformed)
    #
    # assert any(isinstance(tf, ToString) for n, tf in tf2.transformer.steps)
    # assert any(isinstance(tf, HTMLRemover) for n,
    # tf in tf2.transformer.steps)
    # assert isinstance(tf2.transformer.steps[0][1], ToString)
    # assert isinstance(tf2.transformer.steps[1][1], HTMLRemover)
    # assert isinstance(tf2.transformer.steps[2][1], DataSeriesSelector)
    # assert isinstance(tf2.transformer.steps[3][1], TfidfVectorizer)


def test_smart_text_wrong_intent():
    import pandas as pd

    from foreshadow.smart import TextEncoder

    X1 = pd.DataFrame(data=["1", "4", "a", "a"], columns=["col1"])

    manager = CacheManager()
    manager[AcceptedKey.INTENT, "col1"] = IntentType.TEXT

    encoder1 = TextEncoder(cache_manager=manager)

    with pytest.raises(ValueError) as e:
        encoder1.fit(X1)
        assert "empty vocabulary" in str(e)
