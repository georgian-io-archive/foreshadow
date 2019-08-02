import pytest


@pytest.mark.skip("to be removed and replaced with new intent equiv.")
def test_call_classmethod_from_BaseIntent():
    from foreshadow.concrete import BaseIntent

    with pytest.raises(TypeError) as e1:
        BaseIntent.to_string()

    with pytest.raises(TypeError) as e2:
        BaseIntent.priority_traverse()

    with pytest.raises(TypeError) as e3:
        BaseIntent.is_intent([])

    with pytest.raises(TypeError) as e4:
        BaseIntent.column_summary([])

    assert "cannot be called on BaseIntent" in str(e1.value)
    assert "cannot be called on BaseIntent" in str(e2.value)
    assert "cannot be called on BaseIntent" in str(e3.value)
    assert "cannot be called on BaseIntent" in str(e4.value)


@pytest.mark.skip("to be removed and replaced with new intent equiv.")
def test_mock_subclass_missing_abstract_methods():
    from foreshadow.concrete import BaseIntent

    with pytest.raises(NotImplementedError) as e:

        class TestIntent(BaseIntent):
            pass

    assert "has not implemented abstract methods" in str(e.value)


@pytest.mark.skip("to be removed and replaced with new intent equiv.")
def test_mock_subclass_missing_children():
    from foreshadow.concrete import BaseIntent

    with pytest.raises(NotImplementedError) as e:

        class TestIntent(BaseIntent):
            @classmethod
            def is_intent(cls, df):
                return True

            @classmethod
            def column_summary(cls, df):
                return {}

            dtype = "TEST"

    assert "Subclass must define" in str(e.value)


@pytest.mark.skip("to be removed and replaced with new intent equiv.")
def test_mock_subclass_missing_single_pipeline_template():
    from foreshadow.concrete import BaseIntent

    with pytest.raises(NotImplementedError) as e:

        class TestIntent(BaseIntent):
            @classmethod
            def is_intent(cls, df):
                return True

            @classmethod
            def column_summary(cls, df):
                return {}

            dtype = "TEST"
            children = []

    assert "Subclass must define" in str(e.value)


@pytest.mark.skip("to be removed and replaced with new intent equiv.")
def test_mock_subclass_missing_multi_pipeline_template():
    from foreshadow.concrete import BaseIntent

    with pytest.raises(NotImplementedError) as e:

        class TestIntent(BaseIntent):
            @classmethod
            def is_intent(cls, df):
                return True

            @classmethod
            def column_summary(cls, df):
                return {}

            dtype = "TEST"
            children = []
            single_pipeline_template = []

    assert "Subclass must define" in str(e.value)


@pytest.mark.skip("to be removed and replaced with new intent equiv.")
def test_valid_mock_subclass():
    from foreshadow.concrete import _unregister_intent
    from foreshadow.concrete import BaseIntent

    class TestIntent(BaseIntent):
        @classmethod
        def is_intent(cls, df):
            return True

        @classmethod
        def column_summary(cls, df):
            return {}

        dtype = "TEST"
        children = []
        single_pipeline_template = []
        multi_pipeline_template = []

    _ = TestIntent()
    _unregister_intent(TestIntent.__name__)


@pytest.mark.skip("to be removed and replaced with new intent equiv.")
def test_to_string():
    from foreshadow.concrete import _unregister_intent
    from foreshadow.concrete import BaseIntent

    class TestIntent(BaseIntent):
        dtype = "TEST"
        children = ["TestIntent1", "TestIntent2"]
        single_pipeline_template = []
        multi_pipeline_template = []

        @classmethod
        def is_intent(cls, df):
            return True

        @classmethod
        def column_summary(cls, df):
            return {}

    class TestIntent1(TestIntent):
        dtype = "TEST"
        children = ["TestIntent11", "TestIntent12"]
        single_pipeline_template = []
        multi_pipeline_template = []

        @classmethod
        def is_intent(cls, df):
            return True

        @classmethod
        def column_summary(cls, df):
            return {}

    class TestIntent2(TestIntent):
        dtype = "TEST"
        children = []
        single_pipeline_template = []
        multi_pipeline_template = []

        @classmethod
        def is_intent(cls, df):
            return True

        @classmethod
        def column_summary(cls, df):
            return {}

    class TestIntent11(TestIntent1):
        dtype = "TEST"
        children = []
        single_pipeline_template = []
        multi_pipeline_template = []

        @classmethod
        def is_intent(cls, df):
            return True

        @classmethod
        def column_summary(cls, df):
            return {}

    class TestIntent12(TestIntent1):
        dtype = "TEST"
        children = []
        single_pipeline_template = []
        multi_pipeline_template = []

        @classmethod
        def is_intent(cls, df):
            return True

        @classmethod
        def column_summary(cls, df):
            return {}

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


@pytest.mark.skip("to be removed and replaced with new intent equiv.")
def test_priority_traverse():
    from foreshadow.concrete import _unregister_intent
    from foreshadow.concrete import BaseIntent

    class TestIntent(BaseIntent):
        dtype = "TEST"
        children = ["TestIntent1", "TestIntent2"]
        single_pipeline_template = []
        multi_pipeline_template = []

        @classmethod
        def is_intent(cls, df):
            return True

        @classmethod
        def column_summary(cls, df):
            return {}

    class TestIntent1(TestIntent):
        dtype = "TEST"
        children = ["TestIntent11", "TestIntent12"]
        single_pipeline_template = []
        multi_pipeline_template = []

        @classmethod
        def is_intent(cls, df):
            return True

        @classmethod
        def column_summary(cls, df):
            return {}

    class TestIntent2(TestIntent):
        dtype = "TEST"
        children = []
        single_pipeline_template = []
        multi_pipeline_template = []

        @classmethod
        def is_intent(cls, df):
            return True

        @classmethod
        def column_summary(cls, df):
            return {}

    class TestIntent11(TestIntent1):
        dtype = "TEST"
        children = []
        single_pipeline_template = []
        multi_pipeline_template = []

        @classmethod
        def is_intent(cls, df):
            return True

        @classmethod
        def column_summary(cls, df):
            return {}

    class TestIntent12(TestIntent1):
        dtype = "TEST"
        children = []
        single_pipeline_template = []
        multi_pipeline_template = []

        @classmethod
        def is_intent(cls, df):
            return True

        @classmethod
        def column_summary(cls, df):
            return {}

    class_list = [
        TestIntent,
        TestIntent1,
        TestIntent11,
        TestIntent12,
        TestIntent2,
    ]

    assert class_list == list(TestIntent.priority_traverse())
    _unregister_intent(list(map(lambda x: x.__name__, class_list)))
