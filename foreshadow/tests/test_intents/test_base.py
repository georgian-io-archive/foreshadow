import pytest


def test_call_classmethod_from_BaseIntent():
    from foreshadow.intents.base import BaseIntent

    with pytest.raises(TypeError) as e1:
        BaseIntent.to_string()

    with pytest.raises(TypeError) as e2:
        BaseIntent.priority_traverse()

    with pytest.raises(TypeError) as e3:
        BaseIntent.is_intent()

    assert "cannot be called on BaseIntent" in str(e1.value)
    assert "cannot be called on BaseIntent" in str(e2.value)
    assert "cannot be called on BaseIntent" in str(e3.value)


def test_mock_subclass_missing_is_intent():
    from foreshadow.intents.base import BaseIntent

    with pytest.raises(NotImplementedError) as e:

        class TestIntent(BaseIntent):
            pass

    assert "has not implemented abstract methods is_intent" in str(e.value)


def test_mock_subclass_missing_dtype():
    from foreshadow.intents.base import BaseIntent

    with pytest.raises(NotImplementedError) as e:

        class TestIntent(BaseIntent):
            @classmethod
            def is_intent(cls, df):
                return True

    assert "Subclass must define" in str(e.value)


def test_mock_subclass_missing_children():
    from foreshadow.intents.base import BaseIntent

    with pytest.raises(NotImplementedError) as e:

        class TestIntent(BaseIntent):
            @classmethod
            def is_intent(cls, df):
                return True

            dtype = "TEST"

    assert "Subclass must define" in str(e.value)


def test_mock_subclass_missing_single_pipeline():
    from foreshadow.intents.base import BaseIntent

    with pytest.raises(NotImplementedError) as e:

        class TestIntent(BaseIntent):
            @classmethod
            def is_intent(cls, df):
                return True

            dtype = "TEST"
            children = []

    assert "Subclass must define" in str(e.value)


def test_mock_subclass_missing_multi_pipeline():
    from foreshadow.intents.base import BaseIntent

    with pytest.raises(NotImplementedError) as e:

        class TestIntent(BaseIntent):
            @classmethod
            def is_intent(cls, df):
                return True

            dtype = "TEST"
            children = []
            single_pipeline = []

    assert "Subclass must define" in str(e.value)


def test_valid_mock_subclass():
    from foreshadow.intents.registry import _unregister_intent
    from foreshadow.intents.base import BaseIntent

    class TestIntent(BaseIntent):
        @classmethod
        def is_intent(cls, df):
            return True

        dtype = "TEST"
        children = []
        single_pipeline = []
        multi_pipeline = []

    t = TestIntent()
    _unregister_intent(TestIntent.__name__)


def test_to_string():
    from foreshadow.intents.registry import _unregister_intent
    from foreshadow.intents.base import BaseIntent

    class TestIntent(BaseIntent):
        dtype = "TEST"
        children = ["TestIntent1", "TestIntent2"]
        single_pipeline = []
        multi_pipeline = []

        @classmethod
        def is_intent(cls, df):
            return True

    class TestIntent1(TestIntent):
        dtype = "TEST"
        children = ["TestIntent11", "TestIntent12"]
        single_pipeline = []
        multi_pipeline = []

        @classmethod
        def is_intent(cls, df):
            return True

    class TestIntent2(TestIntent):
        dtype = "TEST"
        children = []
        single_pipeline = []
        multi_pipeline = []

        @classmethod
        def is_intent(cls, df):
            return True

    class TestIntent11(TestIntent1):
        dtype = "TEST"
        children = []
        single_pipeline = []
        multi_pipeline = []

        @classmethod
        def is_intent(cls, df):
            return True

    class TestIntent12(TestIntent1):
        dtype = "TEST"
        children = []
        single_pipeline = []
        multi_pipeline = []

        @classmethod
        def is_intent(cls, df):
            return True

    class_list = [
        "TestIntent",
        "TestIntent1",
        "TestIntent2",
        "TestIntent11",
        "TestIntent12",
    ]
    assert TestIntent.to_string() == (
        "TestIntent\n\tTestIntent1\n\t\tTestIntent11\n\t\tTestIntent12\n\t"
        "TestIntent2\n"
    )
    _unregister_intent(class_list)


def test_priority_traverse():
    from foreshadow.intents.registry import _unregister_intent
    from foreshadow.intents.base import BaseIntent

    class TestIntent(BaseIntent):
        dtype = "TEST"
        children = ["TestIntent1", "TestIntent2"]
        single_pipeline = []
        multi_pipeline = []

        @classmethod
        def is_intent(cls, df):
            return True

    class TestIntent1(TestIntent):
        dtype = "TEST"
        children = ["TestIntent11", "TestIntent12"]
        single_pipeline = []
        multi_pipeline = []

        @classmethod
        def is_intent(cls, df):
            return True

    class TestIntent2(TestIntent):
        dtype = "TEST"
        children = []
        single_pipeline = []
        multi_pipeline = []

        @classmethod
        def is_intent(cls, df):
            return True

    class TestIntent11(TestIntent1):
        dtype = "TEST"
        children = []
        single_pipeline = []
        multi_pipeline = []

        @classmethod
        def is_intent(cls, df):
            return True

    class TestIntent12(TestIntent1):
        dtype = "TEST"
        children = []
        single_pipeline = []
        multi_pipeline = []

        @classmethod
        def is_intent(cls, df):
            return True

    class_list = [TestIntent, TestIntent2, TestIntent1, TestIntent12, TestIntent11]
    assert class_list == list(TestIntent.priority_traverse())
    _unregister_intent(list(map(lambda x: x.__name__, class_list)))
