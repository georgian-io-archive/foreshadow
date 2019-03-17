import pytest


def test_unregister():
    from foreshadow.intents.base import BaseIntent
    from foreshadow.intents.registry import _unregister_intent

    class TestIntent(BaseIntent):
        dtype = "TEST"
        children = ["TEST"]
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
        children = ["TEST"]
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
        children = ["TEST"]
        single_pipeline_template = []
        multi_pipeline_template = []

        @classmethod
        def is_intent(cls, df):
            return True

        @classmethod
        def column_summary(cls, df):
            return {}

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


def test_registry_eval():
    from foreshadow.intents.base import BaseIntent
    from foreshadow.intents.registry import _unregister_intent, registry_eval

    class TestIntent(BaseIntent):
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

    assert registry_eval(TestIntent.__name__) is TestIntent
    _unregister_intent(TestIntent.__name__)


def test_samename_subclass():
    from foreshadow.intents.base import BaseIntent
    from foreshadow.intents.registry import _unregister_intent

    class TestIntent(BaseIntent):
        dtype = "TEST"
        children = ["TEST"]
        single_pipeline_template = []
        multi_pipeline_template = []

        @classmethod
        def is_intent(cls, df):
            return True

        @classmethod
        def column_summary(cls, df):
            return {}

    with pytest.raises(TypeError) as e:

        class TestIntent(BaseIntent):  # noqa: F811
            dtype = "TEST"
            children = ["TEST"]
            single_pipeline_template = []
            multi_pipeline_template = []

            @classmethod
            def is_intent(cls, df):
                return True

            @classmethod
            def column_summary(cls, df):
                return {}

    _unregister_intent(TestIntent.__name__)
    assert str(e.value) == (
        "Intent already exists in registry, use a different name"
    )


def test_invalid_transfomer_template_defenition_length():
    from foreshadow.intents.base import BaseIntent
    from foreshadow.transformers.smart import Scaler
    import sklearn

    with pytest.raises(ValueError) as e:

        class TestIntent(BaseIntent):
            dtype = "TEST"
            children = ["TEST"]
            single_pipeline_template = [("s1", Scaler)]
            multi_pipeline_template = []

            @classmethod
            def is_intent(cls, df):
                return True

            @classmethod
            def column_summary(cls, df):
                return {}

    with pytest.raises(ValueError) as e2:

        class TestIntent2(BaseIntent):
            dtype = "TEST"
            children = ["TEST"]
            single_pipeline_template = []
            multi_pipeline_template = [("s1", sklearn.decomposition.PCA)]

            @classmethod
            def is_intent(cls, df):
                return True

            @classmethod
            def column_summary(cls, df):
                return {}

    assert str(e.value) == ("Malformed transformer entry in template")
    assert str(e2.value) == ("Malformed transformer entry in template")


def test_invalid_transfomer_template_defenition_bad_defenition():
    from foreshadow.intents.base import BaseIntent

    with pytest.raises(ValueError) as e:

        class TestIntent(BaseIntent):
            dtype = "TEST"
            children = ["TEST"]
            single_pipeline_template = [("s1", None, False)]
            multi_pipeline_template = []

            @classmethod
            def is_intent(cls, df):
                return True

            @classmethod
            def column_summary(cls, df):
                return {}

    with pytest.raises(ValueError) as e2:

        class TestIntent2(BaseIntent):
            dtype = "TEST"
            children = ["TEST"]
            single_pipeline_template = []
            multi_pipeline_template = [("s1", (None, {}), False)]

            @classmethod
            def is_intent(cls, df):
                return True

            @classmethod
            def column_summary(cls, df):
                return {}

    assert str(e.value) == ("Malformed transformer entry in template")
    assert str(e2.value) == ("Malformed transformer entry in template")


def test_valid_intent_registration():
    from foreshadow.intents.base import (
        BaseIntent,
        PipelineTemplateEntry,
        TransformerEntry,
    )
    from foreshadow.intents.registry import _unregister_intent
    from foreshadow.transformers.smart import Scaler
    import sklearn

    class TestIntent(BaseIntent):
        dtype = "TEST"
        children = ["TEST"]
        single_pipeline_template = [PipelineTemplateEntry("s1", Scaler, True)]
        multi_pipeline_template = [
            PipelineTemplateEntry(
                "s2",
                TransformerEntry(
                    sklearn.decomposition.PCA, {"n_components": 5}
                ),
                True,
            )
        ]

        @classmethod
        def is_intent(cls, df):
            return True

        @classmethod
        def column_summary(cls, df):
            return {}

    _unregister_intent(TestIntent.__name__)
