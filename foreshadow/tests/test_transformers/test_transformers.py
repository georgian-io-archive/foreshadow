import pytest

from foreshadow.cachemanager import CacheManager
from foreshadow.utils import get_transformer
from foreshadow.utils.testing import get_file_path


def test_transformer_keep_cols():
    import pandas as pd
    from foreshadow.concrete import StandardScaler as CustomScaler

    boston_path = get_file_path("data", "boston_housing.csv")

    df = pd.read_csv(boston_path)

    custom = CustomScaler(keep_columns=True)
    custom_tf = custom.fit_transform(df[["crim"]])

    assert custom_tf.shape[1] == 2


def test_transformer_naming_override():
    from foreshadow.concrete import StandardScaler
    import pandas as pd

    boston_path = get_file_path("data", "boston_housing.csv")

    df = pd.read_csv(boston_path)

    scaler = StandardScaler(name="test", keep_columns=False)
    out = scaler.fit_transform(df[["crim"]])

    assert out.iloc[:, 0].name == "crim"


def test_transformer_naming_default():
    from foreshadow.concrete import StandardScaler
    import pandas as pd

    boston_path = get_file_path("data", "boston_housing.csv")

    df = pd.read_csv(boston_path)

    scaler = StandardScaler(keep_columns=False)
    out = scaler.fit_transform(df[["crim"]])

    assert out.iloc[:, 0].name == "crim"


def test_transformer_parallel_invalid():
    from foreshadow.parallelprocessor import ParallelProcessor

    class InvalidTransformer:
        pass

    t = InvalidTransformer()

    with pytest.raises(TypeError) as e:
        ParallelProcessor([("scaled", t, ["crim", "zn", "indus"])])

    assert str(e.value) == (
        "All estimators should implement fit and "
        "transform. '{}'"
        " (type {}) doesn't".format(t, type(t))
    )


def test_transformer_parallel_empty():
    import pandas as pd
    from foreshadow.parallelprocessor import ParallelProcessor

    boston_path = get_file_path("data", "boston_housing.csv")

    df = pd.read_csv(boston_path)

    proc = ParallelProcessor(
        [
            (
                "scaled",
                ParallelProcessor([("cscale", None, ["crim"])]),
                ["crim", "zn", "indus"],
            )
        ]
    )

    proc.fit(df[[]])
    tf = proc.transform(df[[]])

    assert tf.equals(df[[]])

    tf = proc.fit_transform(df[[]])

    assert tf.equals(df[[]])


def test_transformer_parallel_single_process():
    import pandas as pd

    from foreshadow.parallelprocessor import ParallelProcessor
    from foreshadow.concrete import StandardScaler

    boston_path = get_file_path("data", "boston_housing.csv")

    df = pd.read_csv(boston_path)

    ss = StandardScaler(name="scaled")

    proc = ParallelProcessor(
        [
            (
                "scaled",
                StandardScaler(keep_columns=False),
                ["crim", "zn", "indus"],
            )
        ],
        collapse_index=True,
    )

    ss.fit(df[["crim", "zn", "indus"]])
    proc.fit(df)

    tf = proc.transform(df)
    tf_2 = proc.fit_transform(df)

    assert tf.equals(tf_2)

    tf_norm = ss.transform(df[["crim", "zn", "indus"]])
    tf_others = tf.drop(["crim", "zn", "indus"], axis=1)
    tf_test = pd.concat([tf_norm, tf_others], axis=1)

    assert tf.equals(tf_test)


def test_transformer_multiprocess_dynamic_pipelines_update_cache_manager():
    import pandas as pd

    from foreshadow.parallelprocessor import ParallelProcessor
    from foreshadow.cachemanager import CacheManager
    from foreshadow.smart.intent_resolving import IntentResolver

    boston_path = get_file_path("data", "boston_housing.csv")

    raw_data = pd.read_csv(boston_path)
    df = raw_data[["crim", "zn", "indus"]]

    cs = CacheManager()
    from foreshadow.pipeline import DynamicPipeline

    proc = ParallelProcessor(
        [
            (
                "group1",
                DynamicPipeline(
                    [("resolver", IntentResolver(cache_manager=cs))]
                ),
                ["crim"],
            ),
            (
                "group2",
                DynamicPipeline(
                    [("resolver", IntentResolver(cache_manager=cs))]
                ),
                ["zn"],
            ),
            (
                "group3",
                DynamicPipeline(
                    [("resolver", IntentResolver(cache_manager=cs))]
                ),
                ["indus"],
            ),
        ],
        n_jobs=-1,
        collapse_index=True,
    )

    Xs = proc.fit_transform(df)
    assert Xs.equals(df)
    assert len(cs["intent"]) == len(list(df.columns.values))
    assert (
        cs["intent", "crim"] is not None
        and cs["intent", "zn"] is not None
        and cs["intent", "indus"] is not None
    )


def test_transformer_multiprocess_imputer_not_update_cache_manager():
    import pandas as pd

    from foreshadow.parallelprocessor import ParallelProcessor
    from foreshadow.concrete import StandardScaler
    from foreshadow.cachemanager import CacheManager

    boston_path = get_file_path("data", "boston_housing.csv")

    raw_data = pd.read_csv(boston_path)
    df = raw_data[["crim", "zn", "indus"]]

    cs = CacheManager()
    proc = ParallelProcessor(
        [
            ("group1", StandardScaler(), ["crim"]),
            ("group2", StandardScaler(), ["zn"]),
            ("group3", StandardScaler(), ["indus"]),
        ],
        n_jobs=-1,
        collapse_index=True,
    )

    proc.fit_transform(df)
    assert len(cs["intent"]) == 0
    assert len(cs["domain"]) == 0


def test_transformer_multiprocess_smart_transformers_update_cache_manager():
    import pandas as pd

    from foreshadow.parallelprocessor import ParallelProcessor
    from foreshadow.cachemanager import CacheManager
    from foreshadow.smart.intent_resolving import IntentResolver

    boston_path = get_file_path("data", "boston_housing.csv")

    raw_data = pd.read_csv(boston_path)
    df = raw_data[["crim", "zn", "indus"]]

    cs = CacheManager()

    proc = ParallelProcessor(
        [
            ("group1", IntentResolver(cache_manager=cs), ["crim"]),
            ("group2", IntentResolver(cache_manager=cs), ["zn"]),
            ("group3", IntentResolver(cache_manager=cs), ["indus"]),
        ],
        n_jobs=-1,
        collapse_index=True,
    )

    Xs = proc.fit_transform(df)
    assert Xs.equals(df)
    assert len(cs["intent"]) == len(list(df.columns.values))
    assert (
        cs["intent", "crim"] is not None
        and cs["intent", "zn"] is not None
        and cs["intent", "indus"] is not None
    )


def test_transformer_pipeline():
    import pandas as pd
    import numpy as np

    np.random.seed(1337)

    from foreshadow.concrete import StandardScaler as CustomScaler
    from foreshadow.parallelprocessor import ParallelProcessor

    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import FeatureUnion

    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression

    boston_path = get_file_path("data", "boston_housing.csv")

    df = pd.read_csv(boston_path)

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
                            CustomScaler(keep_columns=False),
                            ["crim", "zn", "indus"],
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

    assert np.array_equal(custom.predict(test), sklearn.predict(test))


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


def test_smarttransformer_instantiate():
    """Instantiating a SmartTransformer should fail"""
    from foreshadow.smart import SmartTransformer

    # Note: cannot use fixture since this is not a subclass of SmartTransformer
    with pytest.raises(TypeError) as e:
        SmartTransformer()

    assert "Can't instantiate abstract class" in str(e.value)


def test_smarttransformer_notsubclassed():
    """SmartTransformer (get_transformer TypeError) not being implemented."""
    from foreshadow.smart.smart import SmartTransformer

    # Note: cannot use fixture since the metaclass implementation sets flags on
    # class definition time.
    class TestSmartTransformer(SmartTransformer):
        pass

    with pytest.raises(TypeError) as e:
        TestSmartTransformer()

    assert "Can't instantiate abstract class" in str(e.value)


def test_smarttransformer_attributeerror(smart_child, mocker):
    import pandas as pd
    from foreshadow.exceptions import TransformerNotFound

    boston_path = get_file_path("data", "boston_housing.csv")

    df = pd.read_csv(boston_path)

    smart = smart_child()
    smart.pick_transformer = mocker.Mock()
    smart.pick_transformer.return_value = "INVALID"

    with pytest.raises(TransformerNotFound):
        smart.fit(df[["crim"]])


def test_smarttransformer_invalidtransformer(smart_child, mocker):
    """Test SmartTransformer initialization """
    import pandas as pd

    boston_path = get_file_path("data", "boston_housing.csv")

    df = pd.read_csv(boston_path)

    class InvalidClass:
        pass

    smart = smart_child()
    smart.pick_transformer = mocker.Mock()
    smart.pick_transformer.return_value = InvalidClass()

    with pytest.raises(ValueError) as e:
        smart.fit(df[["crim"]])

    assert (
        "is neither a scikit-learn Pipeline, FeatureUnion, a "
        "wrapped foreshadow transformer, nor None."
    ) in str(e.value)


def test_smarttransformer_function(smart_child):
    """Test overall SmartTransformer functionality

    Args:
        smart_child: A subclass of SmartTransformer.

    """
    import numpy as np
    import pandas as pd

    from foreshadow.concrete import StandardScaler

    boston_path = get_file_path("data", "boston_housing.csv")

    df = pd.read_csv(boston_path)

    smart = smart_child(cache_manager=CacheManager())
    smart_data = smart.fit_transform(df[["crim"]])

    std = StandardScaler()
    std_data = std.fit_transform(df[["crim"]])

    assert smart_data.equals(std_data)

    smart.fit(df[["crim"]])
    smart_data = smart.transform(df[["crim"]])

    std.fit(df[["crim"]])
    std_data = std.transform(df[["crim"]])

    # TODO, remove when SmartTransformer is no longer wrapped
    # Column names will be different, thus np.allclose() is used
    assert np.allclose(smart_data, std_data)


def test_smarttransformer_fitself(smart_child, mocker):
    """Test that fit returns self.

    This is important so that .fit().transform()

    Args:
        smart_child: A subclass of SmartTransformer

    """
    import pandas as pd

    smart = smart_child(override="Imputer", name="test")
    assert smart.fit(pd.DataFrame([1, 2, 3])) == smart


def test_smarttransformer_function_override(smart_child):
    """Test SmartTransformer override through parameter specification.

    Args:
        smart_child: A subclass of SmartTransformer.

    """
    import numpy as np
    import pandas as pd

    from foreshadow.concrete import Imputer

    boston_path = get_file_path("data", "boston_housing.csv")
    df = pd.read_csv(boston_path)

    smart = smart_child(
        transformer="Imputer", name="impute", cache_manager=CacheManager()
    )
    smart_data = smart.fit_transform(df[["crim"]])

    assert isinstance(smart.transformer, Imputer)
    # assert smart.transformer.name == "impute"
    # not relevant anymore.

    std = Imputer(name="impute")
    std_data = std.fit_transform(df[["crim"]])

    assert smart_data.equals(std_data)

    smart.fit(df[["crim"]])
    smart_data = smart.transform(df[["crim"]])

    std.fit(df[["crim"]])
    std_data = std.transform(df[["crim"]])

    assert std_data.columns[0] == "crim"

    # TODO, remove when SmartTransformer is no longer wrapped
    # Column names will be different, thus np.allclose() is used
    assert np.allclose(smart_data, std_data)


def test_smarttransformer_function_override_invalid(smart_child):
    """Test invalid SmartTransformer override transformer class.

    Args:
        smart_child: A subclass of SmartTransformer.

    """
    from foreshadow.exceptions import TransformerNotFound

    with pytest.raises(TransformerNotFound) as e:
        smart_child(transformer="BAD", cache_manager=CacheManager())

    assert "Could not find transformer BAD in" in str(e.value)


def test_smarttransformer_set_params_override(smart_child):
    """Test invalid SmartTransformer override transformer class.

    Args:
        smart_child: A subclass of SmartTransformer.

    """
    from foreshadow.concrete import StandardScaler

    smart = smart_child(transformer="Imputer")
    smart.set_params(**{"transformer": "StandardScaler"})

    assert isinstance(smart.transformer, StandardScaler)


def test_smarttransformer_set_params_empty(smart_child):
    """Test SmartTransformer empty set_params does not fail.

    Args:
        smart_child: A subclass of SmartTransformer.

    """
    smart = smart_child()
    smart.set_params()

    assert smart.transformer is None


def test_smarttransformer_set_params_default(smart_child):
    """Test SmartTransformer pass-through set_params on selected transformer.

    Args:
        smart_child: A subclass of SmartTransformer.

    """
    smart = smart_child()
    smart.fit([1, 2, 3])
    before = smart.__dict__
    params = smart.get_params()
    smart = smart_child().set_params(**params)

    assert smart.__dict__ == before


def test_smarttransformer_get_params(smart_child):
    """Test SmartTransformer override with init kwargs.

    Args:
        smart_child: A subclass of SmartTransformer.

    """
    cm = CacheManager()
    smart = smart_child(
        transformer="Imputer",
        missing_values="NaN",
        strategy="mean",
        cache_manager=cm,
    )
    smart.fit([1, 2, 3])

    params = smart.get_params()
    print(params)
    assert params == {
        "transformer": smart.transformer,
        "name": None,
        "keep_columns": False,
        "y_var": False,
        "force_reresolve": False,
        "should_resolve": False,
        "cache_manager": cm,
        "check_wrapped": True,
        "transformer__copy": True,
        "transformer__missing_values": "NaN",
        "transformer__strategy": "mean",
        "transformer__verbose": 0,
        "transformer__axis": 0,
    }


def test_smarttransformer_empty_inverse(smart_child):
    """Test SmartTransformer inverse_transform.

    Args:
        smart_child: A subclass of SmartTransformer.

    """
    smart = smart_child(cache_manager=CacheManager())
    smart.fit([1, 2, 10])

    smart.inverse_transform([])


def test_smarttransformer_should_resolve(smart_child, mocker):
    """Test SmartTransformer should_resolve functionality.

    First test if the initial behavior works, only resolves the transformer
    once and does not update chosen transformer on new data.

    Next, test if enabling should resolve allows the transformer choice to be
    updated but only once.

    Lastly, test if force_reresolve allows the transformer choice to be updated
    on each fit.

    Args:
        smart_child: A subclass of SmartTransformer.

    """
    import pandas as pd

    from foreshadow.concrete import StandardScaler, MinMaxScaler

    def pick_transformer(X, y=None, **fit_params):
        data = X.iloc[:, 0]

        if data[0] == 0:
            return StandardScaler()
        else:
            return MinMaxScaler()

    smart = smart_child(cache_manager=CacheManager())
    smart.pick_transformer = pick_transformer

    data1 = pd.DataFrame([0])
    data2 = pd.DataFrame([1])

    smart.fit(data1)
    assert isinstance(smart.transformer, StandardScaler)
    smart.fit(data2)
    assert isinstance(smart.transformer, StandardScaler)

    smart.should_resolve = True
    smart.fit(data2)
    assert isinstance(smart.transformer, MinMaxScaler)
    smart.fit(data1)
    assert isinstance(smart.transformer, MinMaxScaler)

    smart.force_reresolve = True
    smart.fit(data1)
    assert isinstance(smart.transformer, StandardScaler)
    smart.fit(data2)
    assert isinstance(smart.transformer, MinMaxScaler)


def test_sparse_matrix_conversion():
    from foreshadow.concrete import FixedTfidfVectorizer

    corpus = [
        "Hello world!",
        "It's a small world.",
        "Small, incremental steps make progress",
    ]

    tfidf = FixedTfidfVectorizer()

    # This tf generates sparse output by default and if not handled will
    # break pandas wrapper
    tfidf.fit_transform(corpus)


@pytest.mark.parametrize(
    "transformer,input_csv",
    [
        ("StandardScaler", get_file_path("data", "boston_housing.csv")),
        ("OneHotEncoder", get_file_path("data", "boston_housing.csv")),
        ("TfidfTransformer", get_file_path("data", "boston_housing.csv")),
    ],
)
def test_make_pandas_transformer_fit(transformer, input_csv):
    """Test pandas_wrap has initial transformer fit functionality.

        Args:
            transformer: wrapped transformer class name
            input_csv: dataset to test on

    """
    import pandas as pd

    transformer = get_transformer(transformer)()
    df = pd.read_csv(input_csv)
    assert transformer.fit(df) == transformer


@pytest.mark.parametrize(
    "transformer,expected_path",
    [
        ("StandardScaler", "sklearn.preprocessing"),
        ("OneHotEncoder", "category_encoders"),
        ("TfidfTransformer", "sklearn.feature_extraction.text"),
    ],
)
def test_make_pandas_transformer_meta(transformer, expected_path):
    """Test that the wrapped transformer has proper metadata.

    Args:
        transformer: wrapped transformer class name
        expected_path: path to the initial transformer

    Returns:

    """
    expected_class = get_transformer(transformer, source_lib=expected_path)
    transformer = get_transformer(transformer)()

    assert isinstance(transformer, expected_class)  # should remain a subclass
    assert type(transformer).__name__ == expected_class.__name__
    assert transformer.__doc__ == expected_class.__doc__


@pytest.mark.parametrize(
    "transformer,kwargs,sk_path,input_csv",
    [
        (
            "StandardScaler",
            {},
            "sklearn.preprocessing",
            get_file_path("data", "boston_housing.csv"),
        ),
        (
            "OneHotEncoder",
            {},
            "category_encoders",
            get_file_path("data", "boston_housing.csv"),
        ),
        (
            "TfidfTransformer",
            {},
            "sklearn.feature_extraction.text",
            get_file_path("data", "boston_housing.csv"),
        ),
    ],
)
def test_make_pandas_transformer_transform(
    transformer, kwargs, sk_path, input_csv
):
    """Test wrapped transformer has the initial transform functionality.

        Args:
            transformer: wrapped transformer class name
            kwargs: key word arguments for transformer initialization
            sk_path: path to the module containing the wrapped sklearn
                transformer
            input_csv: dataset to test on

    """
    import pandas as pd
    import numpy as np
    from scipy.sparse import issparse

    sk_transformer = get_transformer(transformer, source_lib=sk_path)(**kwargs)
    transformer = get_transformer(transformer)(**kwargs)

    df = pd.read_csv(input_csv)
    crim_df = df[["crim"]]
    transformer.fit(crim_df)
    sk_transformer.fit(crim_df)
    sk_out = sk_transformer.transform(crim_df)
    if issparse(sk_out):
        sk_out = sk_out.toarray()
    assert np.array_equal(transformer.transform(crim_df).values, sk_out)


@pytest.mark.parametrize(
    "transformer,sk_path,input_csv",
    [
        (
            "StandardScaler",
            "sklearn.preprocessing",
            get_file_path("data", "boston_housing.csv"),
        ),
        (
            "TfidfTransformer",
            "sklearn.feature_extraction.text",
            get_file_path("data", "boston_housing.csv"),
        ),
    ],
)
def test_make_pandas_transformer_fit_transform(
    transformer, sk_path, input_csv
):
    """Test wrapped transformer has initial fit_transform functionality.

        Args:
            transformer: wrapped transformer
            sk_path: path to the module containing the wrapped sklearn
                transformer
            input_csv: dataset to test on

    """
    import pandas as pd
    import numpy as np
    from scipy.sparse import issparse

    sk_transformer = get_transformer(transformer, source_lib=sk_path)()
    transformer = get_transformer(transformer)()

    df = pd.read_csv(input_csv)
    crim_df = df[["crim"]]
    sk_out = sk_transformer.fit_transform(crim_df)
    if issparse(sk_out):
        sk_out = sk_out.toarray()
    assert np.array_equal(transformer.fit_transform(crim_df).values, sk_out)


@pytest.mark.parametrize(
    "transformer,sk_path",
    [
        ("StandardScaler", "sklearn.preprocessing"),
        ("TfidfTransformer", "sklearn.feature_extraction.text"),
    ],
)
def test_make_pandas_transformer_init(transformer, sk_path):
    """Test pandas_wrap has initial transformer init functionality.

    Should be able to accept any parameters from the sklearn transformer and
    initialize on the wrapped instance. They should also posses the is_wrapped
    method.

        Args:
            transformer: wrapped transformer
            sk_path: path to the module containing the wrapped sklearn
                transformer
    """
    sk_transformer = get_transformer(transformer, source_lib=sk_path)()
    params = sk_transformer.get_params()
    transformer = get_transformer(transformer)(**params)
