import inspect
import json

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from .transformers import ParallelProcessor
from .intents.intents_registry import registry_eval
from .intents import GenericIntent


class Preprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, from_json=None, **fit_params):

        # Dict where key: str(column_name) value: cls(Intent)
        self.intent_map = {}

        # Dict where key: str(column_name) value: Pipeline
        self.pipeline_map = {}

        # Dict where key: str(column_name) value: list(Intents)
        self.choice_map = {}

        # Dict where key: str(list(colname)) value: Pipeline
        self.multi_column_map = []

        # Dict where key: str(intent_name) value: Pipeline
        self.intent_pipelines = {}
        self.intent_trace = []

        # Final pipeline used for fits and transforms
        self.pipeline = None

        # Fit params overrides
        self.fit_params = fit_params

        # Check if has been fit or not
        self.is_fit = False

        if from_json:
            self.init_json(from_json)

    def get_columns(self, intent):
        return ["{}".format(k)
                for k, v in self.intent_map.items()
                if intent in inspect.getmro(v)]

    def map_intents(self, X_df):
        columns = X_df.columns
        if len(columns) > len(set(columns)):
            raise ValueError("Input dataframe columns must not have the same name.")
        for c in columns:
            col_data = X_df.loc[:, [c]]
            valid_cols = [
                (i, k)
                for i, k in enumerate(GenericIntent.priority_traverse())
                if k.is_intent(col_data)
            ]
            if len(valid_cols) == 0:
                self.choice_map[c] = None
                self.intent_map[c] = None
            else:
                self.choice_map[c] = valid_cols
                self.intent_map[c] = valid_cols[-1][1]

    def build_dependency_order(self):
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

    def map_pipelines(self):
        self.pipeline_map = {**{k: Pipeline(v.single_pipeline)
                                for k, v in self.intent_map.items()
                                if v.__name__ not in self.intent_pipelines.keys()},
                             **{k: self.intent_pipelines[v.__name__].get('single', Pipeline(v.single_pipeline))
                                for k, v in self.intent_map.items()
                                if v.__name__ in self.intent_pipelines.keys()},
                             **self.pipeline_map}

        self.build_dependency_order()

        self.intent_pipelines = {v.__name__: {'multi': Pipeline(v.multi_pipeline
                                                                if len(v.multi_pipeline) > 0
                                                                else [('null', None)]),
                                              **{k: v
                                                 for k, v in self.intent_pipelines.get(v.__name__, {}).items()}}
                                 for v in self.intent_trace}

    def construct_parallel_pipeline(self):
        # Repack pl_map into Parallel Processor
        processors = [(col, [col], pipe) for col, pipe in self.pipeline_map.items() if pipe.steps[0][0] != 'null']
        if len(processors) == 0:
            return None
        return ParallelProcessor(processors)

    def construct_multi_pipeline(self):
        # Repack intent_pipeline dict into list of tuples into Pipeline
        multi_processors = [(val[0], ParallelProcessor([(val[0], val[1], val[2])]))
                            for val in self.multi_column_map
                            if val[2].steps[0][0] != 'null']

        processors = [(intent.__name__, ParallelProcessor([(intent.__name__, self.get_columns(intent),
                                                   self.intent_pipelines[intent.__name__]['multi'])]))
                      for intent in reversed(self.intent_trace)
                      if self.intent_pipelines[intent.__name__]['multi'].steps[0][0] != 'null']

        if len(multi_processors + processors) == 0:
            return None

        return Pipeline(processors + multi_processors)

    def generate_pipeline(self, X):
        self.map_intents(X)
        self.map_pipelines()

        parallel = self.construct_parallel_pipeline()
        multi = self.construct_multi_pipeline()

        pipe = []
        if parallel:
            pipe.append(('single', parallel))
        if multi:
            pipe.append(('multi', multi))

        pipe.append(('collapse', ParallelProcessor([('null', [], None)], collapse_index=True)))

        self.pipeline = Pipeline(pipe)

    def resolve_pipeline(self, pipeline_json):
        pipe = []
        for trans in pipeline_json:
            clsname = trans[0]
            name = trans[1]
            params = trans[2]
            module = __import__('transformers', globals(), locals(), [clsname], 1)
            cls = getattr(module, clsname)

            pipe.append((name, cls(**params)))

        if len(pipe) == 0:
            return Pipeline([('null', None)])

        return Pipeline(pipe)

    def init_json(self, config):
        for k, v in config['columns'].items():
            # Assign custom intent map
            self.intent_map[k] = registry_eval(v[0])

            # Assign custom pipeline map
            if len(v) > 1:
                self.pipeline_map[k] = self.resolve_pipeline(v[1])

        self.multi_column_map = [[v[0], v[1], self.resolve_pipeline(v[2])]
                                 for v in config['postprocess']]

        self.intent_pipelines = {k: {l: self.resolve_pipeline(j)
                                     for l, j in v.items()}
                                 for k, v in config['intents'].items()}

    def get_params(self, deep=True):
        if self.pipeline is None:
            return {}
        return {k: v
                for k, v in self.pipeline.get_params(deep).items()
                if k not in self.fit_params.keys()}

    def set_params(self, **params):
        if self.pipeline is None:
            return
        return self.pipeline.set_params(**{k: v
                                           for k, v in params.items()
                                           if k not in self.fit_params.keys()})

    def serialize_pipeline(self, pipeline):
        return [(type(step[1]).__name__, step[0], step[1].get_params())
                for step in pipeline.steps]

    def serialize(self):
        # Serialize Columns / Intents (Show options)
        json_cols = {k: (self.intent_map[k].__name__,
                         self.serialize_pipeline(self.pipeline_map[k]),
                         [c[1].__name__ for c in self.choice_map[k]])
                      for k in self.intent_map.keys()}

        # Serialize multi-column processors
        json_multi = [[v[0], v[1], self.serialize_pipeline(v[2])]
                      for v in self.multi_column_map]

        # Serialize intent multi processors
        json_intents = {k: {l: self.serialize_pipeline(j)
                            for l, j in v.items() if j.steps[0][0] != 'null'}
                        for k, v in self.intent_pipelines.items()}

        return {'columns': json_cols, 'postprocess': json_multi, 'intents': json_intents}

    def fit(self, X, y=None):
        self.generate_pipeline(X)
        self.is_fit = True
        return self.pipeline.fit(X, y)

    def transform(self, X, y=None):
        if not self.pipeline:
            raise ValueError("Pipeline not fit!")
        return self.pipeline.transform(X)
