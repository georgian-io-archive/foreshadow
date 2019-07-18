import pytest


def test_retrieve_name():
    """Test variable name retrieval."""
    from foreshadow.core.serializers import _retrieve_name

    from sklearn.preprocessing import StandardScaler

    a = 10
    assert _retrieve_name(a) == "a"

    standard = StandardScaler()

    assert _retrieve_name(standard) == "standard"


@pytest.fixture
def base_serializer_setup(mocker):
    """Define subclass of BaseTransformerSerializer for base class testing.

    Args:
        mocker: Required argument for pytest-mock

    Returns:
        tuple: An instance of TestSerializer, test_serialize mock, and
            test_deserialize mock

    """
    from foreshadow.core.serializers import BaseTransformerSerializer

    class TestSerializer(BaseTransformerSerializer):
        OPTIONS = ["test"]
        DEFAULT_OPTION = "test"

    ts_mock = mocker.Mock(return_value={})
    TestSerializer.test_serialize = ts_mock
    td_mock = mocker.Mock()
    TestSerializer.test_deserialize = td_mock

    return TestSerializer(), ts_mock, td_mock


def test_base_transformer_option_routing(base_serializer_setup):
    """Test that the base transformer routes to the correct method calls.

    Args:
        base_serializer_setup: pytest fixture

    """
    ts, ts_mock, td_mock = base_serializer_setup
    ser = ts.serialize()
    ts_mock.assert_called_once()

    ts.deserialize(ser)
    td_mock.assert_called_once()


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
        ts.deserialize({"method": "invalid"})

    assert "Deserialization method must be one of" in str(e2)
