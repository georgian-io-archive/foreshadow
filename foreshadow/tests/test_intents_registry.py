import pytest

from ..intents.intents_base import BaseIntent
from ..intents.intents_registry import get_registry, registry_eval, unregister_intent


def test_unregister():
    class TestIntent(BaseIntent):
        dtype = "TEST"
        children = ["TEST"]

    class TestIntent1(TestIntent):
        dtype = "TEST"
        children = ["TEST"]

    class TestIntent2(TestIntent):
        dtype = "TEST"
        children = ["TEST"]

    unregister_intent(TestIntent2.__name__)
    unregister_intent([TestIntent1.__name__, TestIntent.__name__])


def test_get_registry():
    class TestIntent(BaseIntent):
        dtype = "TEST"
        children = []

    g = get_registry()
    assert g[TestIntent.__name__] is TestIntent
    unregister_intent(TestIntent.__name__)


def test_registry_eval():
    class TestIntent(BaseIntent):
        dtype = "TEST"
        children = []

    assert registry_eval(TestIntent.__name__) is TestIntent
    unregister_intent(TestIntent.__name__)


def test_samename_subclass():
    class TestIntent(BaseIntent):
        dtype = "TEST"
        children = ["TEST"]

    with pytest.raises(TypeError) as e:

        class TestIntent(BaseIntent):
            dtype = "TEST"
            children = ["TEST"]

    unregister_intent(TestIntent.__name__)
    assert str(e.value) == ("Intent already exists in registry, use a different name")
