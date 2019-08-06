import pytest

from foreshadow.utils.testing import get_file_path


# TODO fix this
@pytest.fixture(autouse=True)
def patch_intents(mocker):
    from copy import deepcopy

    from foreshadow.q import (
        BaseIntent,
        PipelineTemplateEntry,
        TransformerEntry,
    )
    from foreshadow.concrete import registry
    from foreshadow.concrete import Imputer, PCA

    _saved_registry = deepcopy(registry._registry)
    registry._registry = {}

    class TestGenericIntent(BaseIntent):
        dtype = "str"
        children = ["TestNumericIntent", "TestIntentOne"]

        single_pipeline_template = []
        multi_pipeline_template = [
            PipelineTemplateEntry(
                "pca",
                TransformerEntry(PCA, {"n_components": 2, "name": "pca"}),
                False,
            )
        ]

        @classmethod
        def is_intent(cls, df):
            return True

        @classmethod
        def column_summary(cls, df):
            return {}

    class TestNumericIntent(TestGenericIntent):
        dtype = "float"
        children = []

        single_pipeline_template = [
            PipelineTemplateEntry(
                "impute",
                TransformerEntry(
                    Imputer, {"strategy": "mean", "name": "impute"}
                ),
                False,
            )
        ]
        multi_pipeline_template = []

        @classmethod
        def is_intent(cls, df):
            return True

        @classmethod
        def column_summary(cls, df):
            return {}

    class TestIntentOne(TestGenericIntent):
        dtype = "str"
        children = ["TestIntentTwo", "TestIntentThree"]

        single_pipeline_template = []
        multi_pipeline_template = []

        @classmethod
        def is_intent(cls, df):
            return False

        @classmethod
        def column_summary(cls, df):
            return {}

    class TestIntentTwo(TestIntentOne):
        dtype = "str"
        children = []

        single_pipeline_template = []
        multi_pipeline_template = []

        @classmethod
        def is_intent(cls, df):
            return False

        @classmethod
        def column_summary(cls, df):
            return {}

    class TestIntentThree(TestIntentOne):
        dtype = "str"
        children = []

        single_pipeline_template = []
        multi_pipeline_template = []

        @classmethod
        def is_intent(cls, df):
            return False

        @classmethod
        def column_summary(cls, df):
            return {}

    mocker.patch(
        "foreshadow.preprocessor.GenericIntent.priority_traverse",
        side_effect=TestGenericIntent.priority_traverse,
    )

    # test runs here
    yield
    # reset registry state
    registry._registry = _saved_registry


@pytest.mark.skip("broken until serialization")
def test_preprocessor_init_empty():
    """Verifies preprocessor initializes with empty values."""

    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    proc = DataPreparer(column_sharer=ColumnSharer())

    assert proc._intent_map == {}
    assert proc._intent_pipelines == {}
    assert proc._intent_trace == []
    assert proc._multi_column_map == []
    assert proc._choice_map == {}


@pytest.mark.skip("broken until serialization")
def test_preprocessor_init_json_intent_map():
    """Loads config from JSON and checks to ensure intent map was populated"""

    import json

    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    test_path = get_file_path(
        "configs", "override_column_intent_pipeline.json"
    )

    proc = DataPreparer(
        column_sharer=ColumnSharer(),
        from_json=json.load(open((test_path), "r")),
    )

    assert "crim" in proc._intent_map.keys()
    assert proc._intent_map["crim"].__name__ == "TestGenericIntent"


@pytest.mark.skip("removed?")
def test_preprocessor_intent_dependency_order():
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    proc = DataPreparer(column_sharer=ColumnSharer())
    proc._intent_map = {
        "1": registry_eval("TestIntentOne"),  # noqa: F821
        "2": registry_eval("TestIntentTwo"),  # noqa: F821
        "3": registry_eval("TestIntentThree"),  # noqa: F821
        "4": registry_eval("TestGenericIntent"),  # noqa: F821
    }

    proc._build_dependency_order()

    print("Intent_trace: ", proc._intent_trace)

    assert [c.__name__ for c in proc._intent_trace] == [
        "TestGenericIntent",
        "TestIntentOne",
        "TestIntentTwo",
        "TestIntentThree",
    ] or [c.__name__ for c in proc._intent_trace] == [
        "TestGenericIntent",
        "TestIntentOne",
        "TestIntentThree",
        "TestIntentTwo",
    ]


@pytest.mark.skip("broken until serialization")
def test_preprocessor_init_json_pipeline_map():
    """Loads config from JSON and checks to ensure pipeline map was populated
    """

    import json

    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer
    from foreshadow.utils import PipelineStep

    test_path = get_file_path(
        "configs", "override_column_intent_pipeline.json"
    )

    dp = DataPreparer(
        column_sharer=ColumnSharer(),
        from_json=json.load(open((test_path), "r")),
    )

    assert "crim" in dp._pipeline_map.keys()
    assert type(dp._pipeline_map["crim"]).__name__ == "SerializablePipeline"
    assert len(dp._pipeline_map["crim"].steps) == 1
    assert dp._pipeline_map["crim"].steps[0][PipelineStep["NAME"]] == "Scaler"

    transformer = dp._pipeline_map["crim"].steps[0][PipelineStep["CLASS"]]

    assert type(transformer).__name__ == "StandardScaler"
    # TODO when this test is replaced, add the new test for name attribute.
    assert not transformer.with_mean


@pytest.mark.skip("broken until serialization")
def test_preprocessor_init_json_multi_pipeline():
    """Loads config from JSON and checks to ensure _multi_column_map was
    populated
    """

    import json

    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer
    from foreshadow.utils import PipelineStep

    test_path = get_file_path("configs", "override_multi_pipeline.json")

    proc = DataPreparer(
        column_sharer=ColumnSharer(), from_json=json.load(open(test_path, "r"))
    )

    assert len(proc._multi_column_map) == 1

    pipe = proc._multi_column_map[0]

    assert len(pipe) == 3
    assert pipe[0] == "pca"
    assert pipe[1] == ["crim", "indus", "nox"]

    obj = pipe[2]

    assert type(obj).__name__ == "SerializablePipeline"
    assert len(obj.steps) == 1
    assert obj.steps[0][PipelineStep["NAME"]] == "PCA"

    transformer = obj.steps[0][PipelineStep["CLASS"]]

    assert type(transformer).__name__ == "PCA"
    # assert hasattr(transformer, "name")
    # TODO when this test is replaced, add the new test for name attribute.
    assert transformer.n_components == 2


@pytest.mark.skip("broken until serialization")
def test_preprocessor_init_json_intent_override_multi():
    """Loads config from JSON and checks to ensure
    multi-pipeline intent maps are populated
    """

    import json

    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer
    from foreshadow.utils import PipelineStep

    test_path = get_file_path("configs", "override_intent_pipeline_multi.json")

    proc = DataPreparer(
        column_sharer=ColumnSharer(),
        from_json=json.load(open((test_path), "r")),
    )

    assert "TestNumericIntent" in proc._intent_pipelines.keys()

    pipes = proc._intent_pipelines["TestNumericIntent"]

    assert "multi" in pipes.keys()

    multi = pipes["multi"]

    assert type(multi).__name__ == "SerializablePipeline"
    assert len(multi.steps) == 1

    step = multi.steps[0]

    assert step[0] == "pca"

    transformer = step[PipelineStep["CLASS"]]

    assert type(transformer).__name__ == "PCA"
    # assert hasattr(transformer, "name")
    # TODO when this test is replaced, add the new test for name attribute.
    assert transformer.n_components == 3


@pytest.mark.skip("broken until serialization")
def test_preprocessor_init_json_intent_override_single():
    """Loads config from JSON and checks to ensure
    single-pipeline intent maps are populated
    """

    import json

    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer
    from foreshadow.utils import PipelineStep

    test_path = get_file_path(
        "configs", "override_intent_pipeline_single.json"
    )

    dp = DataPreparer(
        column_sharer=ColumnSharer(),
        from_json=json.load(open((test_path), "r")),
    )

    assert "TestNumericIntent" in dp._intent_pipelines.keys()

    pipes = dp._intent_pipelines["TestNumericIntent"]

    assert "single" in pipes.keys()

    single = pipes["single"]

    assert type(single).__name__ == "SerializablePipeline"
    assert len(single.steps) == 1

    step = single.steps[0]

    assert step[PipelineStep["NAME"]] == "impute"

    transformer = step[1]

    assert type(transformer).__name__ == "Imputer"
    # assert hasattr(transformer, "name")
    # TODO when this test is replaced, add the new test for name attribute.
    assert transformer.strategy == "mean"


@pytest.mark.skip("Broken until DataPreparer can be serialized.")
def test_preprocessor_fit_map_intents_default():
    """Loads config from JSON and fits preprocessor and ensures config
    intent maps override auto-detect
    """
    import pandas as pd
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    boston_path = get_file_path("data", "boston_housing.csv")

    df = pd.read_csv(boston_path)

    dp = DataPreparer(column_sharer=ColumnSharer())

    dp.fit(df.copy(deep=True))

    assert "crim" in dp._intent_map
    assert dp._intent_map["crim"].__name__ == "TestNumericIntent"


@pytest.mark.skip("broken until serialization")
def test_preprocessor_fit_map_intents_override():
    """Loads config from JSON and fits preprocessor and ensures
    config intent maps override auto-detect
    """

    import json

    import pandas as pd
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    boston_path = get_file_path("data", "boston_housing.csv")
    test_path = get_file_path(
        "configs", "override_column_intent_pipeline.json"
    )

    df = pd.read_csv(boston_path)

    proc_override = DataPreparer(
        column_sharer=ColumnSharer(),
        from_json=json.load(open((test_path), "r")),
    )

    proc_override.fit(df.copy(deep=True))

    assert "crim" in proc_override._intent_map
    assert proc_override._intent_map["crim"].__name__ == "TestGenericIntent"


@pytest.mark.skip("broken until DP can be serialized.")
def test_preprocessor_fit_create_single_pipeline_default():
    """Loads config from JSON and fits preprocessor
    and ensures pipeline maps are overridden
    """
    import pandas as pd
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    boston_path = get_file_path("data", "boston_housing.csv")
    df = pd.read_csv(boston_path)
    cols = list(df)

    dp = DataPreparer(column_sharer=ColumnSharer())
    dp.fit(df.copy(deep=True))

    for c in cols:
        assert c in dp._pipeline_map
        assert type(dp._pipeline_map[c]).__name__ == "SerializablePipeline"

    numeric = registry_eval("TestNumericIntent")  # noqa: F821

    assert str(list(zip(*dp._pipeline_map["crim"].steps))[1]) == str(
        list(zip(*numeric.single_pipeline()))[1]
    )


@pytest.mark.skip("broken until DataPreparer can be serialized")
def test_preprocessor_fit_create_single_pipeline_override_column():
    """Loads config from JSON and fits preprocessor
    and ensures pipeline maps are overridden
    """
    import json

    import pandas as pd
    from foreshadow.preparer import DataPreparer
    from foreshadow.utils import PipelineStep
    from foreshadow.columnsharer import ColumnSharer

    boston_path = get_file_path("data", "boston_housing.csv")
    test_path = get_file_path(
        "configs", "override_column_intent_pipeline.json"
    )

    df = pd.read_csv(boston_path)
    cols = list(df)

    dp = DataPreparer(
        column_sharer=ColumnSharer(),
        from_json=json.load(open((test_path), "r")),
    )
    dp.fit(df.copy(deep=True))

    for c in cols:
        assert c in dp._pipeline_map
        assert type(dp._pipeline_map[c]).__name__ == "SerializablePipeline"

    assert dp._pipeline_map["crim"].steps[0][PipelineStep["NAME"]] == "Scaler"


@pytest.mark.skip("broken until DataPreparer can be serialized.")
def test_preprocessor_fit_create_single_pipeline_override_intent():
    """Loads config from JSON and fits preprocessor
    and ensures pipeline maps are overridden
    """
    import json

    import pandas as pd
    from foreshadow.preparer import DataPreparer
    from foreshadow.utils import PipelineStep

    boston_path = get_file_path("data", "boston_housing.csv")
    test_path = get_file_path(
        "configs", "override_intent_pipeline_single.json"
    )

    df = pd.read_csv(boston_path)
    cols = list(df)

    proc_intent = DataPreparer(from_json=json.load(open((test_path), "r")))

    proc_intent.fit(df.copy(deep=True))

    for c in cols:
        assert c in proc_intent._pipeline_map
        assert (
            type(proc_intent._pipeline_map[c]).__name__
            == "SerializablePipeline"
        )

    assert (
        proc_intent._pipeline_map["crim"].steps[0][PipelineStep["NAME"]]
        == "impute"
    )


@pytest.mark.skip("broken until serialization")
def test_preprocessor_make_empty_pipeline():
    import json

    import pandas as pd
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    boston_path = get_file_path("data", "boston_housing.csv")
    test_path = get_file_path("configs", "empty_pipeline_test.json")

    df = pd.read_csv(boston_path)
    orig = df.copy(deep=True)

    dp = DataPreparer(
        column_sharer=ColumnSharer(), from_json=json.load(open(test_path, "r"))
    )
    dp.fit(df)
    out = dp.transform(df)

    out = out.reindex_axis(sorted(out.columns), axis=1)
    orig = orig.reindex_axis(sorted(orig.columns), axis=1)

    assert out.equals(orig)


@pytest.mark.skip("broken until serialization")
def test_preprocessor_make_pipeline():
    """Loads config from JSON that utilizes all
    functionality of system and verifies successful pipeline completion
    """
    import json

    import pandas as pd
    from collections import Counter
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer
    from foreshadow.utils import PipelineStep

    boston_path = get_file_path("data", "boston_housing.csv")
    test_path = get_file_path("configs", "complete_pipeline_test.json")

    df = pd.read_csv(boston_path)
    dp = DataPreparer(
        column_sharer=ColumnSharer(), from_json=json.load(open(test_path, "r"))
    )

    dp.fit(df)

    assert len(dp.pipeline.steps) == 3

    assert dp.pipeline.steps[0][PipelineStep["NAME"]] == "single"
    assert dp.pipeline.steps[1][PipelineStep["NAME"]] == "multi"
    assert dp.pipeline.steps[2][PipelineStep["NAME"]] == "collapse"

    assert (
        type(dp.pipeline.steps[0][PipelineStep["CLASS"]]).__name__
        == "ParallelProcessor"
    )
    assert (
        type(dp.pipeline.steps[1][PipelineStep["CLASS"]]).__name__
        == "SerializablePipeline"
    )
    assert (
        type(dp.pipeline.steps[2][PipelineStep["CLASS"]]).__name__
        == "ParallelProcessor"
    )

    assert Counter(
        [
            t.steps[0][PipelineStep["NAME"]]
            for t in list(
                zip(
                    *dp.pipeline.steps[0][
                        PipelineStep["CLASS"]
                    ].transformer_list
                )
            )[1]
        ]
    ) == Counter(
        [
            "impute",
            "impute",
            "impute",
            "impute",
            "impute",
            "impute",
            "impute",
            "impute",
            "impute",
            "impute",
            "impute",
            "impute",
            "Scaler",
        ]
    )

    assert len(dp.pipeline.steps[1][PipelineStep["CLASS"]].steps) == 3

    assert (
        dp.pipeline.steps[1][PipelineStep["CLASS"]].steps[0][
            PipelineStep["NAME"]
        ]
        == "TestNumericIntent"
    )
    assert (
        dp.pipeline.steps[1][PipelineStep["CLASS"]]
        .steps[0][PipelineStep["CLASS"]]
        .transformer_list[0][PipelineStep["CLASS"]]
        .steps[0][PipelineStep["NAME"]]
        == "pca"
    )

    assert (
        dp.pipeline.steps[1][PipelineStep["CLASS"]].steps[1][
            PipelineStep["NAME"]
        ]
        == "TestGenericIntent"
    )
    assert str(
        dp.pipeline.steps[1][PipelineStep["CLASS"]]
        .steps[1][PipelineStep["CLASS"]]
        .transformer_list[0][PipelineStep["CLASS"]]
        .steps
    ) == str(
        registry_eval("TestGenericIntent").multi_pipeline()  # noqa: F821
    )

    assert (
        dp.pipeline.steps[1][PipelineStep["CLASS"]].steps[2][
            PipelineStep["NAME"]
        ]
        == "pca"
    )

    assert (
        dp.pipeline.steps[2][PipelineStep["CLASS"]].transformer_list[0][
            PipelineStep["NAME"]
        ]
        == "null"
    )


@pytest.mark.skip("broken until serialization")
def test_preprocessor_fit_transform():  # TODO figure out what this test is
    import json

    import pandas as pd
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    boston_path = get_file_path("data", "boston_housing.csv")
    boston_preprocessed_path = get_file_path(
        "data", "boston_housing_processed.csv"
    )
    test_path = get_file_path("configs", "complete_pipeline_test.json")

    df = pd.read_csv(boston_path)

    truth = pd.read_csv(boston_preprocessed_path, index_col=0)
    dp = DataPreparer(
        column_sharer=ColumnSharer(), from_json=json.load(open(test_path, "r"))
    )
    dp.fit(df)
    out = dp.transform(df)

    assert set([c for l in list(out) for c in l.split("_")]) == set(
        [c for l in list(truth) for c in l.split("_")]
    )


@pytest.mark.skip("broken until serialization")
def test_preprocessor_inverse_transform():
    import numpy as np
    import pandas as pd
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    boston_path = get_file_path("data", "boston_housing.csv")

    df = pd.read_csv(boston_path)
    js = {
        "columns": {
            "medv": {
                "intent": "TestGenericIntent",
                "pipeline": [
                    {
                        "transformer": "StandardScaler",
                        "name": "Scaler",
                        "parameters": {"with_mean": True},
                    }
                ],
            }
        }
    }
    dp = DataPreparer(column_sharer=ColumnSharer(), from_json=js)
    col = df[["medv"]]
    dp.fit(col)

    assert dp.is_linear
    assert np.allclose(dp.inverse_transform(dp.transform(col)), col.values)


@pytest.mark.skip("broken until pytest fixture fixed")
def test_preprocessor_inverse_transform_unfit():
    import pandas as pd
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    dp = DataPreparer(column_sharer=ColumnSharer())

    with pytest.raises(ValueError) as e:
        dp.inverse_transform(pd.DataFrame([1, 2, 3, 4]))

    assert str(e.value) == "Pipeline not fit, cannot transform."


@pytest.mark.skip("broken until serialization")
def test_preprocessor_inverse_transform_multicol():
    import pandas as pd
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    boston_path = get_file_path("data", "boston_housing.csv")

    df = pd.read_csv(boston_path)
    js = {
        "columns": {
            "medv": {
                "intent": "TestGenericIntent",
                "pipeline": [
                    {
                        "transformer": "StandardScaler",
                        "name": "Scaler",
                        "parameters": {"with_mean": True},
                    }
                ],
            }
        }
    }
    dp = DataPreparer(column_sharer=ColumnSharer(), from_json=js)
    col = df[["medv", "crim"]]
    dp.fit(col)
    out = dp.transform(col)

    assert not dp.is_linear

    with pytest.raises(ValueError) as e:
        dp.inverse_transform(out)

    assert str(e.value) == "Pipeline does not support inverse transform!"


@pytest.mark.skip("broken until serialization")
def test_preprocessor_get_params():  # TODO figure out what this test is
    import json
    import pickle

    import pandas as pd
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    boston_path = get_file_path("data", "boston_housing.csv")
    test_path = get_file_path("configs", "test_params.pkl")
    test_path2 = get_file_path("configs", "complete_pipeline_test.json")

    df = pd.read_csv(boston_path)
    # (If you change default configs) or file structure, you will need to
    # verify the outputs are correct manually and regenerate the pickle
    # truth file.
    proc = DataPreparer(
        column_sharer=ColumnSharer(),
        from_json=json.load(open(test_path2, "r")),
    )
    proc.fit(df)

    # (If you change default configs) or file structure, you will need to
    # verify the outputs are correct manually and regenerate the pickle
    # truth file.
    truth = pickle.load(open(test_path, "rb"))

    assert proc.get_params().keys() == truth.keys()


@pytest.mark.skip("broken until serialization")
def test_preprocessor_set_params():  # TODO figure out what this test is
    import json
    import pickle

    import pandas as pd
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    boston_path = get_file_path("data", "boston_housing.csv")
    test_path = get_file_path("configs", "test_params.pkl")
    test_path2 = get_file_path("configs", "complete_pipeline_test.json")

    df = pd.read_csv(boston_path)
    # (If you change default configs) or file structure, you will need to
    # verify the outputs are correct manually and regenerate the pickle
    # truth file.
    params = pickle.load(open(test_path, "rb"))
    proc = DataPreparer(
        column_sharer=ColumnSharer(),
        from_json=json.load(open(test_path2, "r")),
    )

    proc.fit(df)
    proc.set_params(**params)

    assert proc.get_params().keys() == params.keys()


@pytest.mark.skip("broken until serialization")
def test_preprocessor_malformed_json_transformer():
    import json

    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    test_path = get_file_path("configs", "malformed_transformer.json")

    with pytest.raises(ValueError) as e:
        DataPreparer(
            column_sharer=ColumnSharer(),
            from_json=json.load(open((test_path), "r")),
        )

    assert "Malformed transformer" in str(e.value)


@pytest.mark.skip("broken until serialization")
def test_preprocessor_invalid_json_transformer_class():
    import json
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    test_path = get_file_path("configs", "invalid_transformer_class.json")

    with pytest.raises(ValueError) as e:
        DataPreparer(
            column_sharer=ColumnSharer(),
            from_json=json.load(open((test_path), "r")),
        )

    assert str(e.value).startswith("Could not import defined transformer")


@pytest.mark.skip("broken until serialization")
def test_preprocessor_invalid_json_transformer_params():
    import json

    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    test_path = get_file_path("configs", "invalid_transformer_params.json")

    with pytest.raises(ValueError) as e:
        DataPreparer(
            column_sharer=ColumnSharer(),
            from_json=json.load(open((test_path), "r")),
        )

    assert str(e.value).startswith(
        "Params {'BAD': 'INVALID'} invalid for transfomer Imputer"
    )


@pytest.mark.skip("broken until serialization")
def test_preprocessor_get_param_no_pipeline():
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    proc = DataPreparer(column_sharer=ColumnSharer())
    param = proc.get_params()

    assert param == {"from_json": None}


@pytest.mark.skip("broken until serialization")
def test_preprocessor_set_param_no_pipeline():
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    dp = DataPreparer(column_sharer=ColumnSharer())
    params = dp.get_params()
    dp.set_params(**{})
    nparam = dp.get_params()

    assert params == nparam


@pytest.mark.skip("broken until intent are swapped in.")
def test_preprocessor_transform_no_pipeline():
    import pandas as pd
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    boston_path = get_file_path("data", "boston_housing.csv")

    dp = DataPreparer(column_sharer=ColumnSharer())
    df = pd.read_csv(boston_path)
    with pytest.raises(ValueError) as e:
        dp.transform(df)

    assert str(e.value) == "Pipeline not fit!"


@pytest.mark.skip("broken until serialization")
def test_preprocessor_serialize():
    import json

    import pandas as pd
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    boston_path = get_file_path("data", "boston_housing.csv")
    test_path = get_file_path("configs", "test_serialize.json")
    test_path2 = get_file_path("configs", "complete_pipeline_test.json")

    df = pd.read_csv(boston_path)

    truth = json.load(open(test_path, "r"))
    dp = DataPreparer(
        column_sharer=ColumnSharer(),
        from_json=json.load(open(test_path2, "r")),
    )
    dp.fit(df)
    out = dp.serialize()

    assert json.loads(json.dumps(truth)) == json.loads(json.dumps(out))


@pytest.mark.skip("broken until serialization")
def test_preprocessor_continuity():
    import pandas as pd
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    boston_path = get_file_path("data", "boston_housing.csv")

    df = pd.read_csv(boston_path)

    dp = DataPreparer(column_sharer=ColumnSharer())
    dp.fit(df)
    ser = dp.serialize()
    _ = DataPreparer(column_sharer=ColumnSharer(), from_json=ser)

    assert ser == dp.serialize()


@pytest.mark.skip("broken until serialization")
def test_preprocessor_y_var_filtering():
    import pandas as pd
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    boston_path = get_file_path("data", "boston_housing.csv")

    df = pd.read_csv(boston_path)
    y_df = df[["medv"]]

    dp = DataPreparer(column_sharer=ColumnSharer(), y_var=True)

    df_out = dp.fit_transform(y_df)

    assert y_df.equals(df_out)


@pytest.mark.skip("broken until serialization")
def test_preprocessor_summarize():
    import json
    import pandas as pd
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    boston_path = get_file_path("data", "boston_housing.csv")
    test_path = get_file_path("configs", "complete_pipeline_test.json")

    df = pd.read_csv(boston_path)
    dp = DataPreparer(
        column_sharer=ColumnSharer(), from_json=json.load(open(test_path, "r"))
    )
    dp.fit(df)

    expected = {
        "age": {"data": {}, "intent": "TestNumericIntent"},
        "b": {"data": {}, "intent": "TestNumericIntent"},
        "chas": {"data": {}, "intent": "TestNumericIntent"},
        "crim": {"data": {}, "intent": "TestGenericIntent"},
        "dis": {"data": {}, "intent": "TestNumericIntent"},
        "indus": {"data": {}, "intent": "TestGenericIntent"},
        "lstat": {"data": {}, "intent": "TestNumericIntent"},
        "medv": {"data": {}, "intent": "TestNumericIntent"},
        "nox": {"data": {}, "intent": "TestNumericIntent"},
        "ptratio": {"data": {}, "intent": "TestNumericIntent"},
        "rad": {"data": {}, "intent": "TestNumericIntent"},
        "rm": {"data": {}, "intent": "TestNumericIntent"},
        "tax": {"data": {}, "intent": "TestNumericIntent"},
        "zn": {"data": {}, "intent": "TestNumericIntent"},
    }

    assert dp.summarize(df) == expected
