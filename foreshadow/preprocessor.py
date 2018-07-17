import json

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from ..transformers import ParallelProcessor
from ..intents import intents_base


class Preprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, from_json=None, **fit_params):

        # Dict where key: str(column_name) value: cls(Intent)
        self.intent_map = {}

        # Dict where key: str(column_name) value: Pipeline
        self.pipeline_map = {}

        # Dict where key: str(column_name) value: list(Intents)
        self.choice_map = {}

        # Dict where key: str(list(colname)) value: Pipeline
        self.multi_column_map = {}

        # Dict where key: str(intent_name) value: Pipeline
        self.intent_pipelines = {}

        # Final pipeline used for fits and transforms
        self.pipeline = None

        # Fit params overrides
        self.fit_params = fit_params

        if from_json:
            self.init_json(from_json)

    def get_columns(self, intent):
        return [k for k, v in self.intent_map.items() if v == intent]

    def map_intents(self, X):
        #TODO: NOT FINAL VERSION
        self.intent_map = {col: intents_base.get_registry()['NumericIntent']
                           for col in X
                           if col not in self.intent_map.keys()}
        self.choice_map = {col: [intents_base.get_registry()['NumericIntent']]
                           for col in X
                           if col not in self.intent_map.keys()}

    def map_pipelines(self):
        self.pipeline_map = {**{k: v.get_pipeline()
                                for k, v in self.intent_map.items()
                                if not k in self.intent_pipelines.keys()},
                             **{k: self.intent_pipelines[k]["single"]
                                for k, v in self.intent_map.items()
                                if k in self.intent_pipelines.keys()},
                             **self.pipeline_map}

    def construct_parallel_pipeline(self, pl_map):
        # Repack pl_map into Parallel Processor
        processors = [(col, [col], pipe) for col, pipe in pl_map.items()]
        return ParallelProcessor(processors)

    def construct_multi_pipeline(self, int_pipe):
        # Repack intent_pipeline dict into list of tuples into Pipeline
        multi_processors = [ParallelProcessor([(cols.replace(' ','').split(','), pipe)])
                            for cols, pipe in self.multi_column_map.items()]
        processors = [ParallelProcessor([(self.get_columns(intent), pipe['multi'])])
                      for intent, pipe in int_pipe.items()]
        return multi_processors + processors

    def generate_pipeline(self, X):
        self.map_intents(X)
        self.map_pipelines()

        parallel = self.construct_parallel_pipeline(self.pipeline_map)
        multi = self.construct_multi_pipeline(self.intent_pipelines)

        self.pipeline = Pipeline([('single', parallel), ('multi', multi)])

    def resolve_intent(self, intent_str):
        return intents_base.get_registry()[intent_str]

    def resolve_pipeline(self, pipeline_json):
        pipe = []
        for trans in pipeline_json:
            clsname = trans[0]
            name = trans[1]
            params = trans[2]

            module = __import__('..transformers', globals(), locals(), [clsname], 1)
            cls = getattr(module, clsname)

            pipe.append((name, cls(**params)))

        return Pipeline(pipe)

    def init_json(self, config):
        for k, v in config['columns'].items():
            # Assign custom intent map
            self.intent_map[k] = self.resolve_intent(v[0])

            # Assign custom pipeline map
            if len(v) > 1:
                self.pipeline_map[k] = self.resolve_pipeline(v[1])

        for k, v in config['postprocess'].items():
            self.multi_column_map[k] = (v[0], self.resolve_pipeline(v[1]))

        for k, v in config['intents'].items():
            # Assign custom single pipeline
            self.intent_pipelines[k]['single'] = self.resolve_pipeline(v['single'])
            self.intent_pipelines[k]['multi'] = self.resolve_pipeline(v['multi'])

    def get_params(self, deep=True):
        return {k: v
                for k, v in self.pipeline.get_params(deep).items()
                if k not in self.fit_params.keys()}

    def set_params(self, **params):
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
                         self.choice_map[k])
                      for k in self.intent_map.keys()}

        # Serialize multi-column processors
        json_multi = {k: self.serialize_pipeline(v)
                      for k, v in self.multi_column_map.items()}

        # Serialize intent multi processors
        json_intents = {k: {'single': self.serialize_pipeline(v['single']),
                            'multi': self.serialize_pipeline(v['multi'])}
                        for k, v in self.intent_pipelines.items()}

        return {'columns': json_cols, 'postprocess': json_multi, 'intents': json_intents}

    def fit(self, X, y=None):
        self.generate_pipeline()
        return self.pipeline.fit(X, y)

    def transform(self, X, y=None):
        if not self.pipeline:
            raise ValueError("Pipeline not fit!")
        return self.pipeline.transform(X, y)
