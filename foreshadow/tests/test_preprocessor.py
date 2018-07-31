import pytest


def test_preprocessor_init_empty():
    """Verifies that preprocessor object initializes correctly with empty values."""

    from foreshadow.preprocessor import Preprocessor

    proc = Preprocessor()

    assert proc.intent_map == {}
    assert proc.intent_pipelines == {}
    assert proc.intent_trace == []
    assert proc.multi_column_map == []
    assert proc.choice_map == {}


def test_preprocessor_init_json_intent_map():
    """Loads config from JSON and checks to ensure intent map was populated"""

    import json
    from foreshadow.preprocessor import Preprocessor

    proc = Preprocessor(
        from_json=json.load(
            open(
                "./foreshadow/tests/test_configs/override_column_intent_pipeline.json",
                "r",
            )
        )
    )

    assert "crim" in proc.intent_map.keys()
    assert proc.intent_map["crim"].__name__ == "GenericIntent"


def test_preprocessor_intent_dependency_order():

    from foreshadow.preprocessor import Preprocessor
    from foreshadow.intents import GenericIntent

    class MockIntentOne(GenericIntent):
        dtype = "str"
        children = ["MockIntentTwo", "MockIntentThree"]

    class MockIntentTwo(MockIntentOne):
        dtype = "str"
        children = []

    class MockIntentThree(MockIntentOne):
        dtype = "str"
        children = []

    proc = Preprocessor()
    proc.intent_map = {
        "1": MockIntentOne,
        "2": MockIntentTwo,
        "3": MockIntentThree,
        "4": GenericIntent,
    }

    proc._build_dependency_order()

    print(proc.intent_trace)

    assert [c.__name__ for c in proc.intent_trace] == [
        "GenericIntent",
        "MockIntentOne",
        "MockIntentTwo",
        "MockIntentThree",
    ] or [c.__name__ for c in proc.intent_trace] == [
        "GenericIntent",
        "MockIntentOne",
        "MockIntentThree",
        "MockIntentTwo",
    ]


def test_preprocessor_init_json_pipeline_map():
    """Loads config from JSON and checks to ensure pipeline map was populated"""

    import json
    from foreshadow.preprocessor import Preprocessor

    proc = Preprocessor(
        from_json=json.load(
            open(
                "./foreshadow/tests/test_configs/override_column_intent_pipeline.json",
                "r",
            )
        )
    )

    assert "crim" in proc.pipeline_map.keys()
    assert type(proc.pipeline_map["crim"]).__name__ == "Pipeline"
    assert len(proc.pipeline_map["crim"].steps) == 1
    assert proc.pipeline_map["crim"].steps[0][0] == "Scaler"

    transformer = proc.pipeline_map["crim"].steps[0][1]

    assert type(transformer).__name__ == "StandardScaler"
    assert hasattr(transformer, "name")
    assert not transformer.with_mean


def test_preprocessor_init_json_multi_pipeline():
    """Loads config from JSON and checks to ensure multi_column_map was populated"""

    import json
    from foreshadow.preprocessor import Preprocessor

    proc = Preprocessor(
        from_json=json.load(
            open("./foreshadow/tests/test_configs/override_multi_pipeline.json", "r")
        )
    )

    assert len(proc.multi_column_map) == 1

    pipe = proc.multi_column_map[0]

    assert len(pipe) == 3
    assert pipe[0] == "pca"
    assert pipe[1] == ["crim", "indus", "nox"]

    obj = pipe[2]

    assert type(obj).__name__ == "Pipeline"
    assert len(obj.steps) == 1
    assert obj.steps[0][0] == "PCA"

    transformer = obj.steps[0][1]

    assert type(transformer).__name__ == "PCA"
    assert hasattr(transformer, "name")
    assert transformer.n_components == 2


def test_preprocessor_init_json_intent_override_multi():
    """Loads config from JSON and checks to ensure
    multi-pipeline intent maps are populated"""

    import json
    from foreshadow.preprocessor import Preprocessor

    proc = Preprocessor(
        from_json=json.load(
            open(
                "./foreshadow/tests/test_configs/override_intent_pipeline_multi.json",
                "r",
            )
        )
    )

    assert "NumericIntent" in proc.intent_pipelines.keys()

    pipes = proc.intent_pipelines["NumericIntent"]

    assert "multi" in pipes.keys()

    multi = pipes["multi"]

    assert type(multi).__name__ == "Pipeline"
    assert len(multi.steps) == 1

    step = multi.steps[0]

    assert step[0] == "pca"

    transformer = step[1]

    assert type(transformer).__name__ == "PCA"
    assert hasattr(transformer, "name")
    assert transformer.n_components == 3


def test_preprocessor_init_json_intent_override_single():
    """Loads config from JSON and checks to ensure
    single-pipeline intent maps are populated"""

    import json
    from foreshadow.preprocessor import Preprocessor

    proc = Preprocessor(
        from_json=json.load(
            open(
                "./foreshadow/tests/test_configs/override_intent_pipeline_single.json",
                "r",
            )
        )
    )

    assert "NumericIntent" in proc.intent_pipelines.keys()

    pipes = proc.intent_pipelines["NumericIntent"]

    assert "single" in pipes.keys()

    single = pipes["single"]

    assert type(single).__name__ == "Pipeline"
    assert len(single.steps) == 1

    step = single.steps[0]

    assert step[0] == "impute"

    transformer = step[1]

    assert type(transformer).__name__ == "Imputer"
    assert hasattr(transformer, "name")
    assert transformer.strategy == "mean"


def test_preprocessor_fit_map_intents_default():
    """Loads config from JSON and fits preprocessor and ensures config
    intent maps override auto-detect"""
    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    df = pd.read_csv("./foreshadow/tests/test_data/boston_housing.csv")

    proc_default = Preprocessor()

    proc_default.fit(df.copy(deep=True))

    assert "crim" in proc_default.intent_map
    assert proc_default.intent_map["crim"].__name__ == "NumericIntent"


def test_preprocessor_fit_map_intents_override():
    """Loads config from JSON and fits preprocessor and ensures
    config intent maps override auto-detect"""

    import json
    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    df = pd.read_csv("./foreshadow/tests/test_data/boston_housing.csv")

    proc_override = Preprocessor(
        from_json=json.load(
            open(
                "./foreshadow/tests/test_configs/override_column_intent_pipeline.json",
                "r",
            )
        )
    )

    proc_override.fit(df.copy(deep=True))

    assert "crim" in proc_override.intent_map
    assert proc_override.intent_map["crim"].__name__ == "GenericIntent"


def test_preprocessor_fit_create_single_pipeline_default():
    """Loads config from JSON and fits preprocessor
    and ensures pipeline maps are overridden"""

    import pandas as pd
    from foreshadow.preprocessor import Preprocessor
    from foreshadow.intents.intents_registry import registry_eval

    df = pd.read_csv("./foreshadow/tests/test_data/boston_housing.csv")
    cols = list(df)

    proc_default = Preprocessor()
    proc_default.fit(df.copy(deep=True))

    for c in cols:
        assert c in proc_default.pipeline_map
        assert type(proc_default.pipeline_map[c]).__name__ == "Pipeline"

    numeric = registry_eval("NumericIntent")
    assert (
        list(zip(*proc_default.pipeline_map["crim"].steps))[1]
        == list(zip(*numeric.single_pipeline))[1]
    )


def test_preprocessor_fit_create_single_pipeline_override_column():
    """Loads config from JSON and fits preprocessor
    and ensures pipeline maps are overridden"""

    import json
    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    df = pd.read_csv("./foreshadow/tests/test_data/boston_housing.csv")
    cols = list(df)

    proc_column = Preprocessor(
        from_json=json.load(
            open(
                "./foreshadow/tests/test_configs/override_column_intent_pipeline.json",
                "r",
            )
        )
    )
    proc_column.fit(df.copy(deep=True))

    for c in cols:
        assert c in proc_column.pipeline_map
        assert type(proc_column.pipeline_map[c]).__name__ == "Pipeline"

    assert proc_column.pipeline_map["crim"].steps[0][0] == "Scaler"


def test_preprocessor_fit_create_single_pipeline_override_intent():
    """Loads config from JSON and fits preprocessor
    and ensures pipeline maps are overridden"""

    import json
    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    df = pd.read_csv("./foreshadow/tests/test_data/boston_housing.csv")
    cols = list(df)

    proc_intent = Preprocessor(
        from_json=json.load(
            open(
                "./foreshadow/tests/test_configs/override_intent_pipeline_single.json",
                "r",
            )
        )
    )

    proc_intent.fit(df.copy(deep=True))

    for c in cols:
        assert c in proc_intent.pipeline_map
        assert type(proc_intent.pipeline_map[c]).__name__ == "Pipeline"

    assert proc_intent.pipeline_map["crim"].steps[0][0] == "impute"


def test_preprocessor_make_empty_pipeline():

    import json
    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    df = pd.read_csv("./foreshadow/tests/test_data/boston_housing.csv")
    orig = df.copy(deep=True)

    proc = Preprocessor(
        from_json=json.load(
            open("./foreshadow/tests/test_configs/empty_pipeline_test.json", "r")
        )
    )
    proc.fit(df)
    out = proc.transform(df)

    out = out.reindex_axis(sorted(out.columns), axis=1)
    orig = orig.reindex_axis(sorted(orig.columns), axis=1)

    assert out.equals(orig)


def test_preprocessor_make_pipeline():
    """Loads config from JSON that utilizes all
    functionality of system and verifies successful pipeline completion"""

    import json
    import pandas as pd
    from collections import Counter
    from foreshadow.preprocessor import Preprocessor
    from foreshadow.intents.intents_registry import registry_eval

    df = pd.read_csv("./foreshadow/tests/test_data/boston_housing.csv")
    proc = Preprocessor(
        from_json=json.load(
            open("./foreshadow/tests/test_configs/complete_pipeline_test.json", "r")
        )
    )

    print(proc.pipeline_map)
    print(proc.intent_map)

    proc.fit(df)

    assert len(proc.pipeline.steps) == 3

    assert proc.pipeline.steps[0][0] == "single"
    assert proc.pipeline.steps[1][0] == "multi"
    assert proc.pipeline.steps[2][0] == "collapse"

    assert type(proc.pipeline.steps[0][1]).__name__ == "ParallelProcessor"
    assert type(proc.pipeline.steps[1][1]).__name__ == "Pipeline"
    assert type(proc.pipeline.steps[2][1]).__name__ == "ParallelProcessor"

    assert Counter(
        [
            t.steps[0][0]
            for t in list(zip(*proc.pipeline.steps[0][1].transformer_list))[2]
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

    assert len(proc.pipeline.steps[1][1].steps) == 3

    assert proc.pipeline.steps[1][1].steps[0][0] == "NumericIntent"
    assert (
        proc.pipeline.steps[1][1].steps[0][1].transformer_list[0][2].steps[0][0]
        == "pca"
    )

    assert proc.pipeline.steps[1][1].steps[1][0] == "GenericIntent"
    assert (
        proc.pipeline.steps[1][1].steps[1][1].transformer_list[0][2].steps
        == registry_eval("GenericIntent").multi_pipeline
    )

    assert proc.pipeline.steps[1][1].steps[2][0] == "pca"

    assert proc.pipeline.steps[2][1].transformer_list[0][0] == "null"


def test_preprocessor_fit_transform():

    import json
    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    df = pd.read_csv("./foreshadow/tests/test_data/boston_housing.csv")

    truth = pd.read_csv(
        "./foreshadow/tests/test_data/boston_housing_processed.csv", index_col=0
    )
    proc = Preprocessor(
        from_json=json.load(
            open("./foreshadow/tests/test_configs/complete_pipeline_test.json", "r")
        )
    )
    proc.fit(df)
    out = proc.transform(df)

    assert set([c for l in list(out) for c in l.split("_")]) == set(
        [c for l in list(truth) for c in l.split("_")]
    )


def test_preprocessor_get_params():

    import json
    import pickle
    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    df = pd.read_csv("./foreshadow/tests/test_data/boston_housing.csv")
    truth = pickle.load(open("./foreshadow/tests/test_configs/tests_params.pkl", "rb"))
    proc = Preprocessor(
        from_json=json.load(
            open("./foreshadow/tests/test_configs/complete_pipeline_test.json", "r")
        )
    )
    proc.fit(df)

    assert proc.get_params().keys() == truth.keys()


def test_preprocessor_set_params():

    import json
    import pickle
    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    df = pd.read_csv("./foreshadow/tests/test_data/boston_housing.csv")
    params = pickle.load(open("./foreshadow/tests/test_configs/tests_params.pkl", "rb"))
    proc = Preprocessor(
        from_json=json.load(
            open("./foreshadow/tests/test_configs/complete_pipeline_test.json", "r")
        )
    )
    proc.fit(df)

    out = proc.set_params(**params)

    assert out == proc.pipeline


def test_preprocessor_malformed_json_transformer():

    import json
    from foreshadow.preprocessor import Preprocessor

    with pytest.raises(ValueError) as e:
        Preprocessor(
            from_json=json.load(
                open("./foreshadow/tests/test_configs/malformed_transformer.json", "r")
            )
        )

    assert str(e.value).startswith("Malformed transformer")


def test_preprocessor_invalid_json_transformer_class():

    import json
    from foreshadow.preprocessor import Preprocessor

    with pytest.raises(ValueError) as e:
        Preprocessor(
            from_json=json.load(
                open(
                    "./foreshadow/tests/test_configs/invalid_transformer_class.json",
                    "r",
                )
            )
        )

    assert str(e.value).startswith("Could not import defined transformer")


def test_preprocessor_invalid_json_transformer_params():

    import json
    from foreshadow.preprocessor import Preprocessor

    with pytest.raises(ValueError) as e:
        Preprocessor(
            from_json=json.load(
                open(
                    "./foreshadow/tests/test_configs/invalid_transformer_params.json",
                    "r",
                )
            )
        )

    print(str(e.value))
    assert str(e.value).startswith("JSON Configuration is malformed:")


def test_preprocessor_get_param_no_pipeline():

    from foreshadow.preprocessor import Preprocessor

    proc = Preprocessor()
    with pytest.raises(ValueError) as e:
        proc.get_params()

    assert str(e.value) == "Pipeline not fit!"


def test_preprocessor_set_param_no_pipeline():

    from foreshadow.preprocessor import Preprocessor

    proc = Preprocessor()
    with pytest.raises(ValueError) as e:
        proc.set_params(**{})

    assert str(e.value) == "Pipeline not fit!"


def test_preprocessor_transform_no_pipeline():

    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    proc = Preprocessor()
    df = pd.read_csv("./foreshadow/tests/test_data/boston_housing.csv")
    with pytest.raises(ValueError) as e:
        proc.transform(df)

    assert str(e.value) == "Pipeline not fit!"


def test_preprocessor_serialize():

    import json
    import pandas as pd
    from foreshadow.preprocessor import Preprocessor

    df = pd.read_csv("./foreshadow/tests/test_data/boston_housing.csv")

    truth = json.load(open("./foreshadow/tests/test_configs/test_serialize.json", "r"))
    proc = Preprocessor(
        from_json=json.load(
            open("./foreshadow/tests/test_configs/complete_pipeline_test.json", "r")
        )
    )
    proc.fit(df)
    out = proc.serialize()

    print(json.dumps(truth))
    print(json.dumps(out))

    assert json.loads(json.dumps(truth)) == json.loads(json.dumps(out))
