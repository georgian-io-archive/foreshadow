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


def test_mock_subclass_missing_parent():
    with pytest.raises(NotImplementedError) as e:

        class TestIntent(BaseIntent):
            intent = "TEST"
            dtype = "TEST"

    assert str(e.value) == (
        "Subclass must define cls.parent attribute.\nThis attribute should define the "
        "parent of the intent."
    )


def test_mock_subclass_missing_children():
    with pytest.raises(NotImplementedError) as e:

        class TestIntent(BaseIntent):
            intent = "TEST"
            dtype = "TEST"
            parent = "TEST"

    assert str(e.value) == (
        "Subclass must define cls.children attribute.\nThis attribute should define the"
        " children of the intent."
    )


def test_mock_subclass_missing_single_pipeline_spec():
    with pytest.raises(NotImplementedError) as e:

        class TestIntent(BaseIntent):
            intent = "TEST"
            dtype = "TEST"
            parent = "TEST"
            children = ["TEST"]

    assert str(e.value) == (
        "Subclass must define cls.single_pipeline_spec attribute.\nThis attribute "
        "should define the sklearn single pipeline spec of the intent."
    )


def test_mock_subclass_missing_multi_pipeline_spec():
    with pytest.raises(NotImplementedError) as e:

        class TestIntent(BaseIntent):
            intent = "TEST"
            dtype = "TEST"
            parent = "TEST"
            children = ["TEST"]
            single_pipeline_spec = "TEST"

    assert str(e.value) == (
        "Subclass must define cls.multi_pipeline_spec attribute.\nThis attribute should"
        " define the sklearn multi pipeline spec of the intent."
    )


def test_mock_subclass_missing_single_pipeline_and_unregister():
    with pytest.raises(NotImplementedError) as e:

        class TestIntent(BaseIntent):
            intent = "TEST"
            dtype = "TEST"
            parent = "TEST"
            children = ["TEST"]
            single_pipeline_spec = "TEST"
            multi_pipeline_spec = "TEST"

            def __init__(self):
                pass

        # must unregister before instantiation in this case
        unregister_intent(TestIntent.__name__)
        t = TestIntent()

    assert str(e.value) == (
        "Subclass initialize self.single_pipeline to self.single_pipeline_spec"
    )


def test_mock_subclass_missing_multi_pipeline():
    with pytest.raises(NotImplementedError) as e:

        class TestIntent(BaseIntent):
            intent = "TEST"
            dtype = "TEST"
            parent = "TEST"
            children = ["TEST"]
            single_pipeline_spec = "TEST"
            multi_pipeline_spec = "TEST"

            def __init__(self):
                self.single_pipeline = self.single_pipeline_spec

        unregister_intent(TestIntent.__name__)
        t = TestIntent()

    assert str(e.value) == (
        "Subclass initialize self.multi_pipeline to self.multi_pipeline_spec"
    )


def test_valid_mock_subclass():
    class TestIntent(BaseIntent):
        intent = "TEST"
        dtype = "TEST"
        parent = "TEST"
        children = ["TEST"]
        single_pipeline_spec = "TEST"
        multi_pipeline_spec = "TEST"

        def __init__(self):
            self.single_pipeline = self.single_pipeline_spec
            self.multi_pipeline = self.multi_pipeline_spec

    unregister_intent(TestIntent.__name__)
    t = TestIntent()


def test_samename_subclass():
    class TestIntent(BaseIntent):
        intent = "TEST"
        dtype = "TEST"
        parent = "TEST"
        children = ["TEST"]
        single_pipeline_spec = "TEST"
        multi_pipeline_spec = "TEST"

        def __init__(self):
            self.single_pipeline = self.single_pipeline_spec  # pragma: no cover
            self.multi_pipeline = self.multi_pipeline_spec  # pragma: no cover

    with pytest.raises(TypeError) as e:

        class TestIntent(BaseIntent):
            intent = "TEST"
            dtype = "TEST"
            parent = "TEST"
            children = ["TEST"]
            single_pipeline_spec = "TEST"
            multi_pipeline_spec = "TEST"

            def __init__(self):
                self.single_pipeline = self.single_pipeline_spec  # pragma: no cover
                self.multi_pipeline = self.multi_pipeline_spec  # pragma: no cover

    unregister_intent(TestIntent.__name__)
    assert str(e.value) == ("Intent already exists in registry, use a different name")


def test_get_registry():
    class TestIntent(BaseIntent):
        intent = "TEST"
        dtype = "TEST"
        parent = "TEST"
        children = ["TEST"]
        single_pipeline_spec = "TEST"
        multi_pipeline_spec = "TEST"

        def __init__(self):
            self.single_pipeline = self.single_pipeline_spec  # pragma: no cover
            self.multi_pipeline = self.multi_pipeline_spec  # pragma: no cover

    g = get_registry()
    assert g[TestIntent.__name__] is TestIntent
    unregister_intent(TestIntent.__name__)
