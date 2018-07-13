import pytest

from ..intents.intents_base import (
    IntentRegistry,
    BaseIntent,
    get_registry,
    unregister_intent,
)


def test_instantiate_base():
    with pytest.raises(TypeError) as e:
        b = BaseIntent()
    assert str(e.value) == "BaseIntent may not be instantiated"


def test_mock_subclass_missing_intent():

    with pytest.raises(NotImplementedError) as e:

        class TestIntent(BaseIntent):
            pass

    assert str(e.value) == (
        "Subclass must define cls.intent attribute.\nThis attribute should define the "
        "name of the intent."
    )


def test_mock_subclass_missing_dtype():
    with pytest.raises(NotImplementedError) as e:

        class TestIntent(BaseIntent):
            intent = "TEST"

    assert str(e.value) == (
        "Subclass must define cls.dtype attribute.\nThis attribute should define the "
        "dtype of the intent."
    )


def test_mock_subclass_missing_children():
    with pytest.raises(NotImplementedError) as e:

        class TestIntent(BaseIntent):
            intent = "TEST"
            dtype = "TEST"

    assert str(e.value) == (
        "Subclass must define cls.children attribute.\nThis attribute should define the"
        " children of the intent."
    )


def test_valid_mock_subclass():
    class TestIntent(BaseIntent):
        intent = "TEST"
        dtype = "TEST"
        children = ["TEST"]

    unregister_intent(TestIntent.__name__)
    t = TestIntent()


def test_samename_subclass():
    class TestIntent(BaseIntent):
        intent = "TEST"
        dtype = "TEST"
        children = ["TEST"]

    with pytest.raises(TypeError) as e:

        class TestIntent(BaseIntent):
            intent = "TEST"
            dtype = "TEST"
            children = ["TEST"]

    unregister_intent(TestIntent.__name__)
    assert str(e.value) == ("Intent already exists in registry, use a different name")


def test_get_registry():
    class TestIntent(BaseIntent):
        intent = "TEST"
        dtype = "TEST"
        children = ["TEST"]

    g = get_registry()
    assert g[TestIntent.__name__] is TestIntent
    unregister_intent(TestIntent.__name__)
