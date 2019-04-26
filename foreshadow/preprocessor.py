import inspect
from copy import deepcopy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from foreshadow.intents import GenericIntent
from foreshadow.intents.registry import registry_eval
from foreshadow.transformers.base import ParallelProcessor
from foreshadow.utils import PipelineStep, check_df


class Preprocessor(BaseEstimator, TransformerMixin):
    """Serves as a self-contained feature engineering tool. Part of
       Foreshadow suite.

    Implements the sklearn transformer interface and can be used in Pipelines
    and in conjunction with the Foreshadow class.

    Fits to a pandas dataframe and matches columns to a corresponding Intent
    class that contains pipelines neccesary to preprocess the information in
    the feature. Then constructs an sklearn pipeline to execute those
    pipelines across the dataframe.

    Optionally takes in JSON configuration file to override internals
    decision making.

    Parameters:
        from_json: Dictionary representing JSON config file (See docs for more)
        y_var: Boolean that indicates the processing of a response variable

    Attributes:
        pipeline: Internal representation of sklearn pipeline. Can be
                  exported and act independently of Preprocessor object.
        is_fit: Boolean representing the fit state of the internals pipeline.

    """

    def __init__(self, from_json=None, y_var=False, **fit_params):
        self.fit_params = fit_params
        self.from_json = from_json
        self.y_var = y_var
        self._initialize()

    def _initialize(self):
        self._intent_map = {}
        self._pipeline_map = {}
        self._choice_map = {}
        self._multi_column_map = []
        self._intent_pipelines = {}
        self._intent_trace = []
        self.pipeline = None
        self.is_fit = False
        self.is_linear = False
        self._init_json()

    def _get_columns(self, intent):
        """
        Iterates columns in intent_map. If intent or intent in its hierarchy
        matches `intent` then it is appended to a list which is returned.

        Effectively returns all columns related to an intent.

        """
        return [
            "{}".format(k)
            for k, v in self._intent_map.items()
            if intent in inspect.getmro(v)
        ]

    def _map_intents(self, X_df):
        """
        Iterates coluns in dataframe. For each column, the intent tree is
        traversed and the best-match is returned. This key value pair is added
        to a dictionary. Any current values in the intent map are allowed to
        override any new values, this allows the JSON configuration to override
        the automatic system.
        """
        temp_map = {}
        columns = X_df.columns
        # Iterate columns
        for c in columns:
            if c in self._intent_map:
                # column is already mapped to an intent, no need to do the traverse here
                self._choice_map[c] = [(0, self._intent_map[c])]
            else:
                col_data = X_df.loc[:, [c]]
                # Traverse intent tree
                valid_cols = [
                    (i, k)
                    for i, k in enumerate(GenericIntent.priority_traverse())
                    if k.is_intent(col_data)
                ]
                self._choice_map[c] = valid_cols
                temp_map[c] = valid_cols[-1][1]

        # Set intent map with override
        self._intent_map = {**temp_map, **self._intent_map}

    def _build_dependency_order(self):
        """
        Creates the order in which multi_pipelines need to be executed. They
        are executed using a "bottom up" technique.
        """
        # Dict of intent orders (distance from root node)
        intent_order = {}
        # Iterates intents in use
        for intent in set(self._intent_map.values()):

            # Skip if already in order dict
            if intent.__name__ in intent_order.keys():
                continue

            # Use slice to remove own class and object and BaseIntent
            # superclass
            parents = inspect.getmro(intent)[1:-2]
            # Determine order using class hierarchy
            order = len(parents)

            # Add to order dict
            intent_order[intent.__name__] = order

            # Iterate parents
            for p in parents:
                # Skip if parent is already in dict
                if p.__name__ in intent_order.keys():
                    break
                # Decrement order (went up a level)
                order -= 1
                # Add to order dict
                intent_order[p.__name__] = order

        # User order dict to sort all active intents
        intent_list = [(n, i) for i, n in intent_order.items()]
        intent_list.sort(key=lambda x: x[0])
        # Evaluate intents into class objects (stored in dict as strings)
        self._intent_trace = [registry_eval(i) for n, i in intent_list]

    def _map_pipelines(self):
        """
        Creates single and multi-column pipelines
        """

        # Create single pipeline map
        self._pipeline_map = {
            # Creates pipeline object from intent single_pipeline attribute
            **{
                k: Pipeline(deepcopy(v.single_pipeline(self.y_var)))
                for k, v in self._intent_map.items()
                if v.__name__ not in self._intent_pipelines.keys()
                and len(v.single_pipeline(self.y_var)) > 0
            },
            # Extracts already resolved single pipelines from JSON intent
            # overrides
            **{
                k: deepcopy(
                    self._intent_pipelines[v.__name__].get(
                        "single",
                        Pipeline(
                            v.single_pipeline(self.y_var)
                            if len(v.single_pipeline(self.y_var)) > 0
                            else [("null", None)]
                        ),
                    )
                )
                for k, v in self._intent_map.items()
                if v.__name__ in self._intent_pipelines.keys()
            },
            # Column-level pipeline overrides (highest priority)
            **self._pipeline_map,
        }

        # Determine order of multi pipeline execution
        self._build_dependency_order()

        # Build multi pipelines
        self._intent_pipelines = {
            # Iterate intents to execute
            v.__name__: {
                # Fetch multi pipeline from Intent class
                "multi": Pipeline(
                    deepcopy(v.multi_pipeline(self.y_var))
                    if len(v.multi_pipeline(self.y_var)) > 0
                    else [("null", None)]
                ),
                "single": Pipeline(
                    deepcopy(v.single_pipeline(self.y_var))
                    if len(v.single_pipeline(self.y_var)) > 0
                    else [("null", None)]
                ),
                # Extract multi pipeline from JSON config (highest priority)
                **{
                    k: v
                    for k, v in self._intent_pipelines.get(
                        v.__name__, {}
                    ).items()
                },
            }
            for v in self._intent_trace
        }

    def _construct_parallel_pipeline(self):
        """Convert column:pipeline into Parallel Processor"""
        processors = [
            (col, pipe, [col])
            for col, pipe in self._pipeline_map.items()
            if pipe.steps[0][0] != "null"
        ]
        if len(processors) == 0:
            return None
        return ParallelProcessor(processors)

    def _construct_multi_pipeline(self):
        """Repack intent_pipeline dict into list of tuples into Pipeline"""

        # Extract pipelines from postprocess section of JSON config
        multi_processors = [
            (val[0], ParallelProcessor([(val[0], val[2], val[1])]))
            for val in self._multi_column_map
            if val[2].steps[0][0] != "null"
        ]

        # Construct multi pipeline from intent trace and intent pipeline
        # dictionary
        processors = [
            (
                intent.__name__,
                ParallelProcessor(
                    [
                        (
                            intent.__name__,
                            self._intent_pipelines[intent.__name__]["multi"],
                            self._get_columns(intent),
                        )
                    ]
                ),
            )
            for intent in reversed(self._intent_trace)
            if self._intent_pipelines[intent.__name__]["multi"].steps[0][
                PipelineStep["NAME"]
            ]
            != "null"
        ]

        if len(multi_processors + processors) == 0:
            return None

        # Return pipeline with multi_pipeline transformers and postprocess
        # transformers
        return Pipeline(processors + multi_processors)

    def _construct_linear_pipeline(self, X):
        """Get single pipeline from pipeline map"""
        return self._pipeline_map.get(X.columns[0], Pipeline([("null", None)]))

    def _generate_pipeline(self, X):
        """Constructs the final internal pipeline to be used"""

        # Parse JSON config and populate intent_map and pipeline_map
        self._initialize()

        # Map intents to columns
        self._map_intents(X)

        # Map pipelines to columns
        self._map_pipelines()

        # If a single column construct a simple linear pipeline
        if len(X.columns) == 1:
            self.pipeline = self._construct_linear_pipeline(X)
            self.is_linear = True

        # Else construct a full pipeline
        else:

            # Per-column single pipelines
            parallel = self._construct_parallel_pipeline()

            # Multi pipelines
            multi = self._construct_multi_pipeline()

            # Verify null pipeline isn't added
            pipe = []
            if parallel:
                pipe.append(("single", parallel))
            if multi:
                pipe.append(("multi", multi))

            # Ensure output of pipeline has a single index
            pipe.append(
                (
                    "collapse",
                    ParallelProcessor(
                        [("null", None, [])], collapse_index=True
                    ),
                )
            )

            self.pipeline = Pipeline(pipe)

    def _init_json(self):
        """Load and parse JSON config"""

        config = self.from_json
        if config is None:
            return

        try:
            if "y_var" in config.keys():
                self.y_var = config["y_var"]
            # Parse columns section
            if "columns" in config.keys():
                # Iterate columns
                for k, v in config["columns"].items():
                    # Assign custom intent map
                    self._intent_map[k] = registry_eval(v["intent"])

                    # Assign custom pipeline map
                    if "pipeline" in v.keys():
                        self._pipeline_map[k] = resolve_pipeline(v["pipeline"])

            # Resolve postprocess section into a list of pipelines
            if "postprocess" in config.keys():
                self._multi_column_map = [
                    [v["name"], v["columns"], resolve_pipeline(v["pipeline"])]
                    for v in config["postprocess"]
                    if validate_pipeline(v)
                ]

            # Resolve intents section into a dictionary of intents and
            # pipelines
            if "intents" in config.keys():
                self._intent_pipelines = {
                    k: {l: resolve_pipeline(j) for l, j in v.items()}
                    for k, v in config["intents"].items()
                }

        except KeyError as e:
            raise ValueError(
                "JSON Configuration is malformed: {}".format(str(e))
            )
        except ValueError as e:
            raise e

    def get_params(self, deep=True):
        if self.pipeline is None:
            return {"from_json": self.from_json}
        return {
            "from_json": self.from_json,
            **self.pipeline.get_params(deep=deep),
        }

    def set_params(self, **params):

        self.from_json = params.pop("from_json", self.from_json)
        self._init_json()

        if self.pipeline is None:
            return

        self.pipeline.set_params(**params)

    def serialize(self):
        """Serialized internals arguments and logic.

        Creates a python dictionary that represents the processes used to
        transform the dataframe. This can be exported to a JSON file and
        modified in order to change pipeline behavior.

        Returns:
            Dictionary configuration (See docs for more detail)

        """
        json_cols = {
            k: {
                "intent": self._intent_map[k].__name__,
                "pipeline": serialize_pipeline(
                    self._pipeline_map.get(k, Pipeline([("null", None)]))
                ),
                "all_matched_intents": [
                    c[1].__name__ for c in self._choice_map[k]
                ],
            }
            for k in self._intent_map.keys()
        }

        # Serialize multi-column processors
        json_multi = [
            {
                "name": v[0],
                "columns": v[1],
                "pipeline": serialize_pipeline(v[2]),
            }
            for v in self._multi_column_map
        ]

        # Serialize intent multi processors
        json_intents = {
            k: {l: serialize_pipeline(j) for l, j in v.items()}
            for k, v in self._intent_pipelines.items()
        }

        return {
            "columns": json_cols,
            "postprocess": json_multi,
            "intents": json_intents,
            "y_var": self.y_var,
        }

    def summarize(self, df):
        """Uses each column's selected intent to generate statistics

            Args:
                df (pandas.DataFrame): The DataFrame to analyze

            Returns: A json dictionary of values with each key representing
                a column and its the value representing the results of that
                intent's column_summary() function
        """
        return {
            k: {
                "intent": self._intent_map[k].__name__,
                "data": self._intent_map[k].column_summary(df[[k]]),
            }
            for k in self._intent_map.keys()
        }

    def fit(self, X, y=None, **fit_params):
        """Fits internal pipeline to X data

        Args:
            X (:obj:`DataFrame <pd.DataFrame>`, :obj:`numpy.ndarray`, list):
                Input data to be transformed

            y (:obj:`DataFrame <pd.DataFrame>`, :obj:`numpy.ndarray`, list):
                Input data to be transformed

        Returns:
            :obj:`Pipeline <sklearn.pipeline.Pipeline>`: Fitted internal
                                                         pipeline

        """
        X = check_df(X)
        y = check_df(y, ignore_none=True)
        self.from_json = fit_params.pop("from_json", self.from_json)
        self._generate_pipeline(X)
        # import pdb
        # pdb.set_trace()
        self.is_fit = True
        return self.pipeline.fit(X, y, **fit_params)

    def transform(self, X):
        """Transforms X using internal pipeline

        Args:
            X (:obj:`DataFrame <pd.DataFrame>`, :obj:`numpy.ndarray`, list):
                Input data to be transformed

        Returns:
            :obj:`pandas.DataFrame`: DataFrame of transformed data

        """
        X = check_df(X)
        if not self.pipeline:
            raise ValueError("Pipeline not fit!")
        return self.pipeline.transform(X)

    def inverse_transform(self, X):
        """Reverses previous transform on X using internal pipeline

        Args:
            X (:obj:`DataFrame <pd.DataFrame>`, :obj:`numpy.ndarray`, list):
                Input data to be transformed

        Returns:
            :obj:`pandas.DataFrame`: DataFrame of inverse transformed data

        """
        X = check_df(X)
        if not self.pipeline or not self.is_fit:
            raise ValueError("Pipeline not fit, cannot transform.")
        if not self.is_linear:
            raise ValueError("Pipeline does not support inverse transform!")
        return self.pipeline.inverse_transform(X)


def serialize_pipeline(pipeline):
    """Serializes sklearn Pipeline object into JSON object for reconstruction.

    Args:
        pipeline (:obj:`sklearn.pipeline.Pipeline`): Pipeline object to
                                                     serialize

    Returns:
        list: JSON serializable object of form ``[cls, name, {**params}]``
    """
    return [
        {
            "transformer": type(step[PipelineStep["CLASS"]]).__name__,
            "name": step[PipelineStep["NAME"]],
            "parameters": step[PipelineStep["CLASS"]].get_params(deep=False),
        }
        for step in pipeline.steps
        if pipeline.steps[0][PipelineStep["NAME"]] != "null"
    ]


def resolve_pipeline(pipeline_json):
    """Deserializes pipeline from JSON into sklearn Pipeline object.

    Args:
        pipeline_json: list of form ``[cls, name, {**params}]``

    Returns:
        :obj:`sklearn.pipeline.Pipeline`: Pipeline based on JSON

    """
    pipe = []

    module_internals = __import__(
        "transformers.internals", globals(), locals(), ["object"], 1
    )
    module_externals = __import__(
        "transformers.externals", globals(), locals(), ["object"], 1
    )
    module_smart = __import__(
        "transformers.smart", globals(), locals(), ["object"], 1
    )

    for trans in pipeline_json:

        try:
            clsname = trans["transformer"]
            name = trans["name"]
            params = trans.get("parameters", {})

        except KeyError:
            raise KeyError(
                "Malformed transformer {} correct syntax is"
                '["transformer": cls, "name": name, "pipeline": '
                "{{**params}}]".format(trans)
            )

        try:
            search_module = (
                module_internals
                if hasattr(module_internals, clsname)
                else (
                    module_externals
                    if hasattr(module_externals, clsname)
                    else module_smart
                )
            )

            cls = getattr(search_module, clsname)

        except Exception:
            raise ValueError(
                "Could not import defined transformer {}".format(clsname)
            )

        try:
            pipe.append((name, cls(**params)))
        except TypeError:
            raise ValueError(
                "Params {} invalid for transfomer {}".format(
                    params, cls.__name__
                )
            )

    if len(pipe) == 0:
        return Pipeline([("null", None)])

    return Pipeline(pipe)


def validate_pipeline(v):
    """
    Validates that a dictionary contains the correct keys for a pipline

    Args:
      v: (dict) Pipeline dictionary

    Returns: True if dict is valid pipeline
    """

    return (
        "columns" in v.keys() and "pipeline" in v.keys() and "name" in v.keys()
    )
