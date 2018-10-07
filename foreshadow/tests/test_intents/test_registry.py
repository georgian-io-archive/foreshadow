import pytest

from foreshadow.intents.registry import get_registry, registry_eval, _unregister_intent


def test_unregister():
    from foreshadow.intents.base import BaseIntent
    from foreshadow.intents.registry import _unregister_intent

    class TestIntent(BaseIntent):
        dtype = "TEST"
        children = ["TEST"]

    class TestIntent1(TestIntent):
        dtype = "TEST"
        children = ["TEST"]

    class TestIntent2(TestIntent):
        dtype = "TEST"
        children = ["TEST"]

    _unregister_intent(TestIntent2.__name__)
    _unregister_intent([TestIntent1.__name__, TestIntent.__name__])


def test_unregister_invalid_input():
    from foreshadow.intents.registry import _unregister_intent

    with pytest.raises(ValueError) as e:
        _unregister_intent(123)

    assert str(e.value) == "Input must be either a string or a list of strings"


def test_unregister_intent_does_not_exist():
    from foreshadow.intents.registry import _unregister_intent

    with pytest.raises(ValueError) as e1:
        _unregister_intent("IntentDoesNotExist")

    with pytest.raises(ValueError) as e2:
        _unregister_intent(["IntentDoesNotExist"])

    err_str = "was not found in registry"

    assert err_str in str(e1.value)
    assert err_str in str(e2.value)


def test_get_registry():
    from foreshadow.intents.base import BaseIntent
    from foreshadow.intents.registry import _unregister_intent, get_registry

    class TestIntent(BaseIntent):
        dtype = "TEST"
        children = []

    g = get_registry()
    assert g[TestIntent.__name__] is TestIntent
    _unregister_intent(TestIntent.__name__)


def test_registry_eval():
    from foreshadow.intents.base import BaseIntent
    from foreshadow.intents.registry import _unregister_intent, registry_eval

    class TestIntent(BaseIntent):
        dtype = "TEST"
        children = []

    assert registry_eval(TestIntent.__name__) is TestIntent
    _unregister_intent(TestIntent.__name__)


def test_samename_subclass():
    from foreshadow.intents.base import BaseIntent
    from foreshadow.intents.registry import _unregister_intent

    class TestIntent(BaseIntent):
        dtype = "TEST"
        children = ["TEST"]

    with pytest.raises(TypeError) as e:

        class TestIntent(BaseIntent):
            dtype = "TEST"
            children = ["TEST"]

    _unregister_intent(TestIntent.__name__)
    assert str(e.value) == ("Intent already exists in registry, use a different name")
