import inspect

from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from .transformers import ParallelProcessor
from .intents.intents_registry import registry_eval
from .intents import GenericIntent
from .utils import check_df


class Preprocessor(BaseEstimator, TransformerMixin):
    """Serves as a self-contained feature engineering tool. Part of Foreshadow suite.

    Implements the sklearn transformer interface and can be used in Pipelines and in
    conjunction with the Foreshadow class.

    Fits to a pandas dataframe and matches columns to a corresponding Intent class that
    contains pipelines neccesary to preprocess the information in the feature. Then
    constructs an sklearn pipeline to execute those pipelines across the dataframe.

    Optionally takes in JSON configuration file to override internal decision making.

    Attributes:
        intent_map: Dictionary mapping str column name keys to an Intent class
        pipeline_map: Dictionary mapping str column name keys to a Pipeline object
        multi_column_map: Dictionary mapping str transformer name keys to Pipelines
            that act on multiple columns in the data frame.
        intent_pipelines: Dictionary mapping intent class name keys to a dictionary
            containing a 'multi' Pipeline object for collectively processing multiple
            columns of that intent type and a 'single' Pipeline object for independently
            processing individual columns of that intent type.
        intent_trace: List of intents in order of processing depending on Intent
            dependency tree
        pipeline: Internal representation of sklearn pipeline. Can be exported and
            act independently of Preprocessor object.
        is_fit: Boolean representing the fit state of the internal pipeline.

    """

    def __init__(self, from_json=None, **fit_params):
        """Initializes Preprocessor class.

        If from_json is provided then overrides are extracted from the config file
        and used during fitting.

        Args:
            from_json: Dictionary representing JSON config file (See docs for more)

        """

        self.intent_map = {}
        self.pipeline_map = {}
        self.choice_map = {}
        self.multi_column_map = []
        self.intent_pipelines = {}
        self.intent_trace = []
        self.pipeline = None
        self.fit_params = fit_params
        self.is_fit = False
        self.from_json = from_json
        self._init_json()

    def _get_columns(self, intent):
        return [
            "{}".format(k)
            for k, v in self.intent_map.items()
            if intent in inspect.getmro(v)
        ]

    def _map_intents(self, X_df):
        temp_map = {}
        columns = X_df.columns
        for c in columns:
            col_data = X_df.loc[:, [c]]
            valid_cols = [
                (i, k)
                for i, k in enumerate(GenericIntent.priority_traverse())
                if k.is_intent(col_data)
            ]
            self.choice_map[c] = valid_cols
            temp_map[c] = valid_cols[-1][1]
        self.intent_map = {**temp_map, **self.intent_map}

    def _build_dependency_order(self):
        intent_order = {}
        for intent in set(self.intent_map.values()):

            if intent.__name__ in intent_order.keys():
                continue

            # Use slice to remove own class and object and BaseIntent superclass
            parents = inspect.getmro(intent)[1:-2]
            order = len(parents)

            intent_order[intent.__name__] = order
            for p in parents:
                if p.__name__ in intent_order.keys():
                    break
                order -= 1
                intent_order[p.__name__] = order

        intent_list = [(n, i) for i, n in intent_order.items()]
        intent_list.sort(key=lambda x: x[0])
        self.intent_trace = [registry_eval(i) for n, i in intent_list]

    def _map_pipelines(self):
        self.pipeline_map = {
            **{
                k: Pipeline(deepcopy(v.single_pipeline))
                for k, v in self.intent_map.items()
                if v.__name__ not in self.intent_pipelines.keys()
                and len(v.single_pipeline) > 0
            },
            **{
                k: self.intent_pipelines[v.__name__].get(
                    "single", Pipeline(deepcopy(v.single_pipeline) if len(
                        v.single_pipeline) >
                                                            0 else [("null", None)])
                )
                for k, v in self.intent_map.items()
                if v.__name__ in self.intent_pipelines.keys()
            },
            **self.pipeline_map,
        }

        self._build_dependency_order()

        self.intent_pipelines = {
            v.__name__: {
                "multi": Pipeline(
                    deepcopy(v.multi_pipeline) if len(v.multi_pipeline) > 0 else [(
                        "null", None)]
                ),
                **{k: v for k, v in self.intent_pipelines.get(v.__name__, {}).items()},
            }
            for v in self.intent_trace
        }

    def _construct_parallel_pipeline(self):
        # Repack pl_map into Parallel Processor
        processors = [
            (col, [col], pipe)
            for col, pipe in self.pipeline_map.items()
            if pipe.steps[0][0] != "null"
        ]
        if len(processors) == 0:
            return None
        return ParallelProcessor(processors)

    def _construct_multi_pipeline(self):
        # Repack intent_pipeline dict into list of tuples into Pipeline
        multi_processors = [
            (val[0], ParallelProcessor([(val[0], val[1], val[2])]))
            for val in self.multi_column_map
            if val[2].steps[0][0] != "null"
        ]

        processors = [
            (
                intent.__name__,
                ParallelProcessor(
                    [
                        (
                            intent.__name__,
                            self._get_columns(intent),
                            self.intent_pipelines[intent.__name__]["multi"],
                        )
                    ]
                ),
            )
            for intent in reversed(self.intent_trace)
            if self.intent_pipelines[intent.__name__]["multi"].steps[0][0] != "null"
        ]

        if len(multi_processors + processors) == 0:
            return None

        return Pipeline(processors + multi_processors)

    def _generate_pipeline(self, X):
        self._init_json()
        self._map_intents(X)
        self._map_pipelines()

        parallel = self._construct_parallel_pipeline()
        multi = self._construct_multi_pipeline()

        pipe = []
        if parallel:
            pipe.append(("single", parallel))
        if multi:
            pipe.append(("multi", multi))

        pipe.append(
            ("collapse", ParallelProcessor([("null", [], None)], collapse_index=True))
        )

        self.pipeline = Pipeline(pipe)

    def _init_json(self):

        config = self.from_json
        if config is None:
            return

        try:

            if "columns" in config.keys():
                for k, v in config["columns"].items():
                    # Assign custom intent map
                    self.intent_map[k] = registry_eval(v[0])

                    # Assign custom pipeline map
                    if len(v) > 1:
                        self.pipeline_map[k] = resolve_pipeline(v[1])

            if "postprocess" in config.keys():
                self.multi_column_map = [
                    [v[0], v[1], resolve_pipeline(v[2])]
                    for v in config["postprocess"]
                    if len(v) >= 3
                ]

            if "intents" in config.keys():
                self.intent_pipelines = {
                    k: {l: resolve_pipeline(j) for l, j in v.items()}
                    for k, v in config["intents"].items()
                }

        except ValueError as e:
            raise e
        except Exception as e:
            raise ValueError("JSON Configuration is malformed: {}".format(str(e)))

    def get_params(self, deep=True):
        if self.pipeline is None:
            return {'from_json': self.from_json}
        return {'from_json': self.from_json,  **self.pipeline.get_params(deep=deep)}

    def set_params(self, **params):

        self.from_json = params.pop('from_json', self.from_json)
        self._init_json()

        if self.pipeline is None:
            return

        self.pipeline.set_params(**params)

    def serialize(self):
        """Serialized internal arguments and logic.

        Creates a python dictionary that represents the processes used to transform
        the dataframe. This can be exported to a JSON file and modified in order to
        change pipeline behavior.

        Returns:
            Dictionary configuration (See docs for more detail)

        """
        json_cols = {
            k: (
                self.intent_map[k].__name__,
                serialize_pipeline(
                    self.pipeline_map.get(k, Pipeline([("null", None)]))
                ),
                [c[1].__name__ for c in self.choice_map[k]],
            )
            for k in self.intent_map.keys()
        }

        # Serialize multi-column processors
        json_multi = [
            [v[0], v[1], serialize_pipeline(v[2])] for v in self.multi_column_map
        ]

        # Serialize intent multi processors
        json_intents = {
            k: {
                l: serialize_pipeline(j)
                for l, j in v.items()
                if j.steps[0][0] != "null"
            }
            for k, v in self.intent_pipelines.items()
        }

        return {
            "columns": json_cols,
            "postprocess": json_multi,
            "intents": json_intents,
        }

    def fit(self, X, y=None, **fit_params):
        """See base class."""
        X = check_df(X)
        y = check_df(y, ignore_none=True)
        self.from_json = fit_params.pop('from_json', self.from_json)
        self._generate_pipeline(X)
        # import pdb
        # pdb.set_trace()
        self.is_fit = True
        return self.pipeline.fit(X, y, **fit_params)

    def transform(self, X):
        """See base class."""
        X = check_df(X)
        if not self.pipeline:
            raise ValueError("Pipeline not fit!")
        return self.pipeline.transform(X)

    def inverse_transform(self, X):
        X = check_df(X)
        if not self.pipeline or not self.is_fit:
            raise ValueError("Pipeline not fit, cannot transform.")
        return self.pipeline.inverse_transform(X)


def serialize_pipeline(pipeline):
    """Serializes sklearn Pipeline object into JSON object for reconstruction.

    Returns:
        List of form [cls, name, {**params}]
    """
    return [
        (type(step[1]).__name__, step[0], step[1].get_params())
        for step in pipeline.steps
        if pipeline.steps[0][0] != "null"
    ]


def resolve_pipeline(pipeline_json):
    """Deserializes pipeline from JSON into sklearn Pipeline object.

    Args:
        pipeline_json: List of form [cls, name, {**params}]

    Returns:
        Sklearn Pipeline object of form Pipeline([(name, cls(**params)), ...])

    """
    pipe = []
    for trans in pipeline_json:

        if len(trans) != 3:
            raise ValueError(
                "Malformed transformer {} correct syntax is"
                "[cls, name, {{**params}}]".format(trans)
            )

        clsname = trans[0]
        name = trans[1]
        params = trans[2]

        try:
            module = __import__("transformers", globals(), locals(), [clsname], 1)
            cls = getattr(module, clsname)
        except Exception as e:
            raise ValueError("Could not import defined transformer {}".format(clsname))

        pipe.append((name, cls(**params)))

    if len(pipe) == 0:
        return Pipeline([("null", None)])

    return Pipeline(pipe)
