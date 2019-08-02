import numpy as np
import pytest


def _assert_dict_equal(d1, d2):
    # drop specific keys that are causing equality checking problems
    drop_keys = ["ngram_range", "_tfidf"]
    [d1.pop(k, None) for k in drop_keys]
    [d2.pop(k, None) for k in drop_keys]

    assert d1 == d2


@pytest.fixture
def base_serializer_setup(mocker):
    """Define subclass of BaseTransformerSerializer for base class testing.

    Args:
        mocker: Required argument for pytest-mock

    Returns:
        tuple: An instance of TestSerializable, test_serialize mock, and
            test_deserialize mock

    """
    from foreshadow.serializers import BaseTransformerSerializer

    class TestSerializable(BaseTransformerSerializer):
        OPTIONS = ["test"]
        DEFAULT_OPTION = "test"

    ts_mock = mocker.Mock(return_value={})
    TestSerializable.test_serialize = ts_mock
    td_mock = mocker.Mock(return_value=TestSerializable())
    TestSerializable.test_deserialize = td_mock

    return TestSerializable(), ts_mock, td_mock


def test_base_transformer_option_routing(base_serializer_setup):
    """Test that the base transformer routes to the correct method calls.

    Args:
        base_serializer_setup: pytest fixture

    """
    ts, ts_mock, td_mock = base_serializer_setup
    ser = ts.serialize()
    ts_mock.assert_called_once()

    assert "_class" in ser
    assert "_method" in ser

    deser = ts.deserialize(ser)
    td_mock.assert_called_once()

    assert isinstance(deser, ts.__class__)


def test_base_transformer_invalid_option(base_serializer_setup):
    """Test that the base transformer raises when an invalid method is passed.

    Args:
        base_serializer_setup: pytest fixture

    """
    ts, *_ = base_serializer_setup

    with pytest.raises(ValueError) as e1:
        ts.serialize(method="invalid")

    assert "Serialization method must be one of" in str(e1)

    with pytest.raises(ValueError) as e2:
        ts.deserialize({"_method": "invalid"})

    assert "Deserialization method must be one of" in str(e2)


@pytest.fixture(
    params=[
        (
            "sklearn.preprocessing",
            "StandardScaler",
            np.arange(10).reshape((-1, 1)),
        ),
        (
            "sklearn.feature_extraction.text",
            "TfidfVectorizer",
            ["hello", "world", "hello world"],
        ),
    ]
)
def concrete_serializer(request, mocker):
    """Define subclass of BaseTransformerSerializer for base class testing.

    Args:
        request: The fixture parameter holders

    Returns:
        tuple: An instance of TestSerializer, test_serialize mock, and
            test_deserialize mock

    """
    from importlib import import_module

    from foreshadow.serializers import ConcreteSerializerMixin

    module, tf_class_name, data = request.param

    tf_class = getattr(import_module(module), tf_class_name)

    class ConcreteSerializable(tf_class, ConcreteSerializerMixin):
        pass

    # Elevate to globals to allow pickling
    globals()["ConcreteSerializable"] = ConcreteSerializable
    ConcreteSerializable.__qualname__ = "ConcreteSerializable"

    concrete = ConcreteSerializable().fit(data)

    return concrete


def test_concrete_default_pathway(concrete_serializer):
    """Test default end to end serialization/deserialization of transformer.

    Args:
        concrete_serializer: pytest fixture that constructs a test transformer.

    """

    ser = concrete_serializer.serialize()
    deser = concrete_serializer.deserialize(ser)

    _assert_dict_equal(concrete_serializer.get_params(), deser.get_params())


def test_concrete_dict_serdeser(concrete_serializer):
    """Test dict (params) serialization/deserialization of transformer.

    Args:
        concrete_serializer: pytest fixture that constructs a test transformer.

    """

    ser = concrete_serializer.serialize(method="dict")
    deser = concrete_serializer.deserialize(ser)
    _assert_dict_equal(concrete_serializer.get_params(), deser.get_params())


def test_concrete_inline_serdeser(concrete_serializer):
    """Test inline (pickle) serialization/deserialization of transformer.

    Args:
        concrete_serializer: pytest fixture that constructs a test transformer.

    """

    ser = concrete_serializer.serialize(method="inline")
    deser = concrete_serializer.deserialize(ser)

    _assert_dict_equal(concrete_serializer.__dict__, deser.__dict__)


def test_concrete_disk_ser(concrete_serializer, mocker):
    """Test disk (pickle, file) serialization/deserialization of transformer.

    Args:
        concrete_serializer: pytest fixture that constructs a test transformer.

    """

    m = mocker.mock_open()
    mocker.patch("builtins.open", m, create=True)
    test_path = "./test/path"
    mocker.patch(
        ("foreshadow.serializers._pickle_cache_path"), return_value=test_path
    )
    ser = concrete_serializer.serialize(method="disk")

    # # Generate a pickle string for the next test, make sure to comment out
    # # the mocks
    # import pickle
    # with open(f'{ser["data"]}', 'rb') as fopen:
    #     ob = pickle.load(fopen)

    m.assert_called_once()
    assert ser["_file_path"] == test_path


def test_concrete_disk_deser(concrete_serializer, mocker):
    """Test disk (pickle, file) serialization/deserialization of transformer.

    Args:
        concrete_serializer: pytest fixture that constructs a test transformer.

    """
    pickle_str = (
        b"\x80\x03cforeshadow.tests.test_core.test_serializer\n"
        b"ConcreteSerializable\nq\x00)\x81q\x01}q\x02(X\t\x00\x00\x00"
        b"with_meanq\x03\x88X\x08\x00\x00\x00with_stdq\x04\x88X\x04\x00\x00"
        b"\x00copyq\x05\x88X\x05\x00\x00\x00mean_q\x06cnumpy.core.multiarray"
        b"\n_reconstruct\nq\x07cnumpy\nndarray\nq\x08K\x00\x85q\tC\x01bq\n"
        b"\x87q\x0bRq\x0c(K\x01K\x01\x85q\rcnumpy\ndtype\nq\x0eX\x02\x00\x00"
        b"\x00f8q\x0fK\x00K\x01\x87q\x10Rq\x11(K\x03X\x01\x00\x00\x00<q\x12NN"
        b"NJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tq\x13b\x89C\x08\x00\x00"
        b"\x00\x00\x00\x00\x12@q\x14tq\x15bX\x0f\x00\x00\x00n_samples_seen_q"
        b"\x16K\nX\x04\x00\x00\x00var_q\x17h\x07h\x08K\x00\x85q\x18h\n\x87q"
        b"\x19Rq\x1a(K\x01K\x01\x85q\x1bh\x11\x89C\x08\x00\x00\x00\x00\x00"
        b"\x80 @q\x1ctq\x1dbX\x06\x00\x00\x00scale_q\x1eh\x07h\x08K\x00\x85q"
        b'\x1fh\n\x87q Rq!(K\x01K\x01\x85q"h\x11\x89C\x08\xf0\xd0b\xa1n\xfa'
        b"\x06@q#tq$bub."
    )
    m = mocker.mock_open(read_data=pickle_str)
    mocker.patch("builtins.open", m, create=True)
    _ = concrete_serializer.deserialize(
        {"_file_path": "./test/path", "_method": "disk"}
    )
    m.assert_called_once()


def test_concrete_disk_serdeser_custom_path(concrete_serializer, tmp_path):
    ser = concrete_serializer.serialize(method="disk", cache_path=tmp_path)
    deser = concrete_serializer.deserialize(ser)

    _assert_dict_equal(concrete_serializer.__dict__, deser.__dict__)


@pytest.mark.parametrize(
    "file_ser_method",
    [("json", "to_json", "from_json"), ("yaml", "to_yaml", "from_yaml")],
)
def test_concrete_json(file_ser_method, concrete_serializer, tmp_path):
    import os

    # TODO: generalize these tests to work with all types of serializations
    file_ext, to_file, from_file = file_ser_method
    to_file = getattr(concrete_serializer, to_file)
    from_file = getattr(concrete_serializer, from_file)

    fpath = os.path.join(tmp_path, "test.json")
    _ = to_file(path=fpath, method="dict")
    deser = from_file(path=fpath)

    _assert_dict_equal(concrete_serializer.get_params(), deser.get_params())


def test_pipeline_serializable_dict_ser(concrete_serializer, mocker):
    from foreshadow.serializers import PipelineSerializerMixin

    p = PipelineSerializerMixin()
    mocker.patch(
        "foreshadow.serializers.PipelineSerializerMixin.get_params",
        return_value={"steps": [("test", concrete_serializer)]},
        create=True,
    )
    mocker.patch(
        "foreshadow.serializers.get_transformer",
        return_value=concrete_serializer.__class__,
    )
    mocker.patch(
        "foreshadow.serializers.PipelineSerializerMixin.__init__",
        return_value=None,
    )

    ser = p.serialize()
    deser = p.deserialize(ser)

    _assert_dict_equal(p.get_params(), deser.get_params())


def test_deserialize_function(concrete_serializer, mocker):
    from foreshadow.serializers import deserialize

    mocker.patch(
        "foreshadow.serializers.get_transformer",
        return_value=concrete_serializer.__class__,
    )
    ser = concrete_serializer.serialize()
    deser = deserialize(ser)

    _assert_dict_equal(concrete_serializer.get_params(), deser.get_params())


def test_concrete_custom_class(concrete_serializer, mocker):
    from foreshadow.serializers import deserialize
    from foreshadow.serializers import _pickle_inline_repr

    mocker.patch(
        "foreshadow.serializers.ConcreteSerializerMixin.pickle_class_def",
        return_value={
            "_pickled_class": _pickle_inline_repr(
                concrete_serializer.__class__
            )
        },
    )
    ser = concrete_serializer.serialize()
    deser = deserialize(ser)

    _assert_dict_equal(concrete_serializer.get_params(), deser.get_params())
