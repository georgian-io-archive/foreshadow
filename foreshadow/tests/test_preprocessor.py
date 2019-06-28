import pytest


@pytest.fixture(autouse=True)
def patch_intents(mocker):
    from copy import deepcopy

    from foreshadow.intents.base import (
        BaseIntent,
        PipelineTemplateEntry,
        TransformerEntry,
    )
    from foreshadow.intents import registry
    from foreshadow.transformers.externals import Imputer, PCA

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


def test_preprocessor_init_empty():
    """Verifies preprocessor initializes with empty values."""

    from foreshadow.preprocessor import Preprocessor

    proc = Preprocessor()

    assert proc._intent_map == {}
    assert proc._intent_pipelines == {}
    assert proc._intent_trace == []
    assert proc._multi_column_map == []
    assert proc._choice_map == {}


def test_preprocessor_init_json_intent_map():
    """Loads config from JSON and checks to ensure intent map was populated"""

    import json
    import os

    from foreshadow.preprocessor import Preprocessor

    test_path = os.path.join(
        os.path.dirname(__file__),
        "test_configs",
        "override_column_intent_pipeline.json",
    )

    proc = Preprocessor(from_json=json.load(open((test_path), "r")))

    assert "crim" in proc._intent_map.keys()
    print(proc._intent_map)
    print(type(proc._intent_map["crim"]))
    assert proc._intent_map["crim"].__name__ == "TestGenericIntent"


def test_preprocessor_intent_dependency_order():
    from foreshadow.preprocessor import Preprocessor
    from foreshadow.intents.registry import registry_eval

    proc = Preprocessor()
    proc._intent_map = {
        "1": registry_eval("TestIntentOne"),
        "2": registry_eval("TestIntentTwo"),
        "3": registry_eval("TestIntentThree"),
        "4": registry_eval("TestGenericIntent"),
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


def test_preprocessor_init_json_pipeline_map():
    """Loads config from JSON and checks to ensure pipeline map was populated
    """

    import json
    import os

    from foreshadow.preprocessor import Preprocessor
    from foreshadow.utils import PipelineStep

    test_path = os.path.join(
        os.path.dirname(__file__),
        "test_configs",
        "override_column_intent_pipeline.json",
    )

    proc = Preprocessor(from_json=json.load(open((test_path), "r")))

    assert "crim" in proc._pipeline_map.keys()
    assert type(proc._pipeline_map["crim"]).__name__ == "Pipeline"
    assert len(proc._pipeline_map["crim"].steps) == 1
    assert (
        proc._pipeline_map["crim"].steps[0][PipelineStep["NAME"]] == "Scaler"
    )

    transformer = proc._pipeline_map["crim"].steps[0][PipelineStep["CLASS"]]

    assert type(transformer).__name__ == "StandardScaler"
    assert hasattr(transformer, "name")
    assert not transformer.with_mean


def test_preprocessor_init_json_multi_pipeline():
    """Loads config from JSON and checks to ensure _multi_column_map was
    populated
    """

    import json
    import os

    from foreshadow.preprocessor import Preprocessor
    from foreshadow.utils import PipelineStep

    test_path = os.path.join(
        os.path.dirname(__file__),
        "test_configs",
        "override_multi_pipeline.json",
    )

    proc = Preprocessor(from_json=json.load(open(test_path, "r")))

    assert len(proc._multi_column_map) == 1

    pipe = proc._multi_column_map[0]

    assert len(pipe) == 3
    assert pipe[0] == "pca"
    assert pipe[1] == ["crim", "indus", "nox"]

    obj = pipe[2]

    assert type(obj).__name__ == "Pipeline"
    assert len(obj.steps) == 1
    assert obj.steps[0][PipelineStep["NAME"]] == "PCA"

    transformer = obj.steps[0][PipelineStep["CLASS"]]

    assert type(transformer).__name__ == "PCA"
    assert hasattr(transformer, "name")
    assert transformer.n_components == 2


def test_preprocessor_init_json_intent_override_multi():
    """Loads config from JSON and checks to ensure
    multi-pipeline intent maps are populated
    """

    import json
    import os

    from foreshadow.preprocessor import Preprocessor
    from foreshadow.utils import PipelineStep

    test_path = os.path.join(
        os.path.dirname(__file__),
        "test_configs",
        "override_intent_pipeline_multi.json",
    )

    proc = Preprocessor(from_json=json.load(open((test_path), "r")))

    assert "TestNumericIntent" in proc._intent_pipelines.keys()

    pipes = proc._intent_pipelines["TestNumericIntent"]

    assert "multi" in pipes.keys()

    multi = pipes["multi"]

    assert type(multi).__name__ == "Pipeline"
    assert len(multi.steps) == 1

    step = multi.steps[0]

    assert step[0] == "pca"

    transformer = step[PipelineStep["CLASS"]]

    assert type(transformer).__name__ == "PCA"
    assert hasattr(transformer, "name")
    assert transformer.n_components == 3


def test_preprocessor_init_json_intent_override_single():
    """Loads config from JSON and checks to ensure
    single-pipeline intent maps are populated
    """

    import json
    import os

    from foreshadow.preprocessor import Preprocessor
    from foreshadow.utils import PipelineStep

    test_path = os.path.join(
        os.path.dirname(__file__),
        "test_configs",
        "override_intent_pipeline_single.json",
    )

    proc = Preprocessor(from_json=json.load(open((test_path), "r")))

    assert "TestNumericIntent" in proc._intent_pipelines.keys()

    pipes = proc._intent_pipelines["TestNumericIntent"]

    assert "single" in pipes.keys()

    single = pipes["single"]

    assert type(single).__name__ == "Pipeline"
    assert len(single.steps) == 1

    step = single.steps[0]

    assert step[PipelineStep["NAME"]] == "impute"

    transformer = step[1]

    assert type(transformer).__name__ == "Imputer"
    assert hasattr(transformer, "name")
    assert transformer.strategy == "mean"


def test_preprocessor_fit_map_intents_default():
    """Loads config from JSON and fits preprocessor and ensures config
    intent maps override auto-detect
    """

    import os

    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )

    df = pd.read_csv(boston_path)

    proc_default = Preprocessor()

    proc_default.fit(df.copy(deep=True))

    assert "crim" in proc_default._intent_map
    assert proc_default._intent_map["crim"].__name__ == "TestNumericIntent"


def test_preprocessor_fit_map_intents_override():
    """Loads config from JSON and fits preprocessor and ensures
    config intent maps override auto-detect
    """

    import json
    import os

    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )
    test_path = os.path.join(
        os.path.dirname(__file__),
        "test_configs",
        "override_column_intent_pipeline.json",
    )

    df = pd.read_csv(boston_path)

    proc_override = Preprocessor(from_json=json.load(open((test_path), "r")))

    proc_override.fit(df.copy(deep=True))

    assert "crim" in proc_override._intent_map
    assert proc_override._intent_map["crim"].__name__ == "TestGenericIntent"


def test_preprocessor_fit_create_single_pipeline_default():
    """Loads config from JSON and fits preprocessor
    and ensures pipeline maps are overridden
    """

    import os

    import pandas as pd
    from foreshadow.preprocessor import Preprocessor
    from foreshadow.intents.registry import registry_eval

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )
    df = pd.read_csv(boston_path)
    cols = list(df)

    proc_default = Preprocessor()
    proc_default.fit(df.copy(deep=True))

    for c in cols:
        assert c in proc_default._pipeline_map
        assert type(proc_default._pipeline_map[c]).__name__ == "Pipeline"

    numeric = registry_eval("TestNumericIntent")

    assert str(list(zip(*proc_default._pipeline_map["crim"].steps))[1]) == str(
        list(zip(*numeric.single_pipeline()))[1]
    )


def test_preprocessor_fit_create_single_pipeline_override_column():
    """Loads config from JSON and fits preprocessor
    and ensures pipeline maps are overridden
    """

    import json
    import os

    import pandas as pd
    from foreshadow.preprocessor import Preprocessor
    from foreshadow.utils import PipelineStep

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )
    test_path = os.path.join(
        os.path.dirname(__file__),
        "test_configs",
        "override_column_intent_pipeline.json",
    )

    df = pd.read_csv(boston_path)
    cols = list(df)

    proc_column = Preprocessor(from_json=json.load(open((test_path), "r")))
    proc_column.fit(df.copy(deep=True))

    for c in cols:
        assert c in proc_column._pipeline_map
        assert type(proc_column._pipeline_map[c]).__name__ == "Pipeline"

    assert (
        proc_column._pipeline_map["crim"].steps[0][PipelineStep["NAME"]]
        == "Scaler"
    )


def test_preprocessor_fit_create_single_pipeline_override_intent():
    """Loads config from JSON and fits preprocessor
    and ensures pipeline maps are overridden
    """

    import json
    import os

    import pandas as pd
    from foreshadow.preprocessor import Preprocessor
    from foreshadow.utils import PipelineStep

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )
    test_path = os.path.join(
        os.path.dirname(__file__),
        "test_configs",
        "override_intent_pipeline_single.json",
    )

    df = pd.read_csv(boston_path)
    cols = list(df)

    proc_intent = Preprocessor(from_json=json.load(open((test_path), "r")))

    proc_intent.fit(df.copy(deep=True))

    for c in cols:
        assert c in proc_intent._pipeline_map
        assert type(proc_intent._pipeline_map[c]).__name__ == "Pipeline"

    assert (
        proc_intent._pipeline_map["crim"].steps[0][PipelineStep["NAME"]]
        == "impute"
    )


def test_preprocessor_make_empty_pipeline():
    import json
    import os

    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )
    test_path = os.path.join(
        os.path.dirname(__file__), "test_configs", "empty_pipeline_test.json"
    )

    df = pd.read_csv(boston_path)
    orig = df.copy(deep=True)

    proc = Preprocessor(from_json=json.load(open(test_path, "r")))
    proc.fit(df)
    out = proc.transform(df)

    out = out.reindex_axis(sorted(out.columns), axis=1)
    orig = orig.reindex_axis(sorted(orig.columns), axis=1)

    assert out.equals(orig)


def test_preprocessor_make_pipeline():
    """Loads config from JSON that utilizes all
    functionality of system and verifies successful pipeline completion
    """

    import json
    import os

    import pandas as pd
    from collections import Counter
    from foreshadow.preprocessor import Preprocessor
    from foreshadow.intents.registry import registry_eval
    from foreshadow.utils import PipelineStep

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )
    test_path = os.path.join(
        os.path.dirname(__file__),
        "test_configs",
        "complete_pipeline_test.json",
    )

    df = pd.read_csv(boston_path)
    proc = Preprocessor(from_json=json.load(open(test_path, "r")))

    proc.fit(df)

    assert len(proc.pipeline.steps) == 3

    assert proc.pipeline.steps[0][PipelineStep["NAME"]] == "single"
    assert proc.pipeline.steps[1][PipelineStep["NAME"]] == "multi"
    assert proc.pipeline.steps[2][PipelineStep["NAME"]] == "collapse"

    assert (
        type(proc.pipeline.steps[0][PipelineStep["CLASS"]]).__name__
        == "ParallelProcessor"
    )
    assert (
        type(proc.pipeline.steps[1][PipelineStep["CLASS"]]).__name__
        == "Pipeline"
    )
    assert (
        type(proc.pipeline.steps[2][PipelineStep["CLASS"]]).__name__
        == "ParallelProcessor"
    )

    assert Counter(
        [
            t.steps[0][PipelineStep["NAME"]]
            for t in list(
                zip(
                    *proc.pipeline.steps[0][
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

    assert len(proc.pipeline.steps[1][PipelineStep["CLASS"]].steps) == 3

    assert (
        proc.pipeline.steps[1][PipelineStep["CLASS"]].steps[0][
            PipelineStep["NAME"]
        ]
        == "TestNumericIntent"
    )
    assert (
        proc.pipeline.steps[1][PipelineStep["CLASS"]]
        .steps[0][PipelineStep["CLASS"]]
        .transformer_list[0][PipelineStep["CLASS"]]
        .steps[0][PipelineStep["NAME"]]
        == "pca"
    )

    assert (
        proc.pipeline.steps[1][PipelineStep["CLASS"]].steps[1][
            PipelineStep["NAME"]
        ]
        == "TestGenericIntent"
    )
    assert str(
        proc.pipeline.steps[1][PipelineStep["CLASS"]]
        .steps[1][PipelineStep["CLASS"]]
        .transformer_list[0][PipelineStep["CLASS"]]
        .steps
    ) == str(registry_eval("TestGenericIntent").multi_pipeline())

    assert (
        proc.pipeline.steps[1][PipelineStep["CLASS"]].steps[2][
            PipelineStep["NAME"]
        ]
        == "pca"
    )

    assert (
        proc.pipeline.steps[2][PipelineStep["CLASS"]].transformer_list[0][
            PipelineStep["NAME"]
        ]
        == "null"
    )


def test_preprocessor_fit_transform():
    import json
    import os

    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )
    boston_preprocessed_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing_processed.csv"
    )
    test_path = os.path.join(
        os.path.dirname(__file__),
        "test_configs",
        "complete_pipeline_test.json",
    )

    df = pd.read_csv(boston_path)

    truth = pd.read_csv(boston_preprocessed_path, index_col=0)
    proc = Preprocessor(from_json=json.load(open(test_path, "r")))
    proc.fit(df)
    out = proc.transform(df)

    assert set([c for l in list(out) for c in l.split("_")]) == set(
        [c for l in list(truth) for c in l.split("_")]
    )


def test_preprocessor_inverse_transform():
    import os

    import numpy as np
    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )

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
    proc = Preprocessor(from_json=js)
    col = df[["medv"]]
    proc.fit(col)

    assert proc.is_linear
    assert np.allclose(
        proc.inverse_transform(proc.transform(col)).values, col.values
    )


def test_preprocessor_inverse_transform_unfit():
    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    proc = Preprocessor()

    with pytest.raises(ValueError) as e:
        proc.inverse_transform(pd.DataFrame([1, 2, 3, 4]))

    assert str(e.value) == "Pipeline not fit, cannot transform."


def test_preprocessor_inverse_transform_multicol():
    import os

    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )

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
    proc = Preprocessor(from_json=js)
    col = df[["medv", "crim"]]
    proc.fit(col)
    out = proc.transform(col)

    assert not proc.is_linear

    with pytest.raises(ValueError) as e:
        proc.inverse_transform(out)

    assert str(e.value) == "Pipeline does not support inverse transform!"


def test_preprocessor_get_params():
    import json
    import os
    import pickle

    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )
    test_path = os.path.join(
        os.path.dirname(__file__), "test_configs", "test_params.pkl"
    )
    test_path2 = os.path.join(
        os.path.dirname(__file__),
        "test_configs",
        "complete_pipeline_test.json",
    )

    df = pd.read_csv(boston_path)
    truth = pickle.load(open(test_path, "rb"))
    proc = Preprocessor(from_json=json.load(open(test_path2, "r")))
    proc.fit(df)

    assert proc.get_params().keys() == truth.keys()


def test_preprocessor_set_params():
    import json
    import os
    import pickle

    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )
    test_path = os.path.join(
        os.path.dirname(__file__), "test_configs", "tests_params.pkl"
    )
    test_path2 = os.path.join(
        os.path.dirname(__file__),
        "test_configs",
        "complete_pipeline_test.json",
    )

    df = pd.read_csv(boston_path)
    params = pickle.load(open(test_path, "rb"))
    proc = Preprocessor(from_json=json.load(open(test_path2, "r")))

    proc.fit(df)
    proc.set_params(**params)

    assert proc.get_params().keys() == params.keys()


def test_preprocessor_malformed_json_transformer():
    import json
    import os

    from foreshadow.preprocessor import Preprocessor

    test_path = os.path.join(
        os.path.dirname(__file__), "test_configs", "malformed_transformer.json"
    )

    with pytest.raises(ValueError) as e:
        Preprocessor(from_json=json.load(open((test_path), "r")))

    assert "Malformed transformer" in str(e.value)


def test_preprocessor_invalid_json_transformer_class():
    import json
    import os

    from foreshadow.preprocessor import Preprocessor

    test_path = os.path.join(
        os.path.dirname(__file__),
        "test_configs",
        "invalid_transformer_class.json",
    )

    with pytest.raises(ValueError) as e:
        Preprocessor(from_json=json.load(open((test_path), "r")))

    assert str(e.value).startswith("Could not import defined transformer")


def test_preprocessor_invalid_json_transformer_params():
    import json
    import os

    from foreshadow.preprocessor import Preprocessor

    test_path = os.path.join(
        os.path.dirname(__file__),
        "test_configs",
        "invalid_transformer_params.json",
    )

    with pytest.raises(ValueError) as e:
        Preprocessor(from_json=json.load(open((test_path), "r")))

    print(str(e.value))
    assert str(e.value).startswith(
        "Params {'BAD': 'INVALID'} invalid for transfomer Imputer"
    )


def test_preprocessor_get_param_no_pipeline():
    from foreshadow.preprocessor import Preprocessor

    proc = Preprocessor()
    param = proc.get_params()

    assert param == {"from_json": None}


def test_preprocessor_set_param_no_pipeline():
    from foreshadow.preprocessor import Preprocessor

    proc = Preprocessor()
    params = proc.get_params()
    proc.set_params(**{})
    nparam = proc.get_params()

    assert params == nparam


def test_preprocessor_transform_no_pipeline():
    import os

    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )

    proc = Preprocessor()
    df = pd.read_csv(boston_path)
    with pytest.raises(ValueError) as e:
        proc.transform(df)

    assert str(e.value) == "Pipeline not fit!"


def test_preprocessor_serialize():
    import json
    import os

    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )
    test_path = os.path.join(
        os.path.dirname(__file__), "test_configs", "test_serialize.json"
    )
    test_path2 = os.path.join(
        os.path.dirname(__file__),
        "test_configs",
        "complete_pipeline_test.json",
    )

    df = pd.read_csv(boston_path)

    truth = json.load(open(test_path, "r"))
    proc = Preprocessor(from_json=json.load(open(test_path2, "r")))
    proc.fit(df)
    out = proc.serialize()

    assert json.loads(json.dumps(truth)) == json.loads(json.dumps(out))


def test_preprocessor_continuity():
    import os

    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )

    df = pd.read_csv(boston_path)

    proc = Preprocessor()
    proc.fit(df)
    ser = proc.serialize()
    _ = Preprocessor(from_json=ser)

    assert ser == proc.serialize()


def test_preprocessor_y_var_filtering():
    import os

    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )

    df = pd.read_csv(boston_path)
    y_df = df[["medv"]]

    proc = Preprocessor(y_var=True)

    df_out = proc.fit_transform(y_df)

    assert y_df.equals(df_out)


def test_preprocessor_summarize():
    import json
    import os

    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    boston_path = os.path.join(
        os.path.dirname(__file__), "test_data", "boston_housing.csv"
    )
    test_path = os.path.join(
        os.path.dirname(__file__),
        "test_configs",
        "complete_pipeline_test.json",
    )

    df = pd.read_csv(boston_path)
    proc = Preprocessor(from_json=json.load(open(test_path, "r")))
    proc.fit(df)

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

    assert proc.summarize(df) == expected
