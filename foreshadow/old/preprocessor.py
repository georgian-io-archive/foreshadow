"""Core preprocessing transformer and helpers."""
# flake8: noqa
# import inspect
# from copy import deepcopy
#
# from foreshadow.base import BaseEstimator, TransformerMixin
#
# from foreshadow.intents import GenericIntent
# from foreshadow.intents.registry import registry_eval
# from foreshadow.core import (
#     ParallelProcessor,
#     SerializablePipeline,
#     SmartTransformer,
# )
# from foreshadow.utils import PipelineStep, check_df, get_transformer
#
#
# class Preprocessor(BaseEstimator, TransformerMixin):
#     """Serves as a self-contained feature engineering tool.
#     Implements the :class:`BaseEstimator <sklearn.base.BaseEstimator>` and
#     :class:`TransformerMixin <sklearn.base.TransformerMixin>` interface and can
#     be used in :class:`Pipeline <sklearn.pipeline.Pipeline>` and in conjunction
#     with the :class:`Foreshadow` class.
#     Fits to a :obj:`DataFrame <pandas.DataFrame>` and matches columns to a
#     corresponding Intent class that contains pipelines necessary to preprocess
#     the information in the feature. Then constructs an sklearn pipeline to
#     execute those pipelines across the dataframe.
#     Optionally takes in JSON configuration file to override internals
#     decision making.
#     Parameters:
#         from_json (dict, optional): Dictionary representing JSON config file
#             (See docs for more)
#         y_var (bool, optional): Boolean that indicates the processing of a
#             response variable
#     Attributes:
#         pipeline (:obj:`Pipeline <sklearn.pipeline.Pipeline>`): Internal
#             representation of sklearn pipeline. Can be exported and act
#             independently of Preprocessor object.
#         is_fit (bool): Boolean representing the fit state of the internals
#             pipeline.
#         is_linear (bool): Boolean representing whether the pipeline can be
#             inverted.
#     """
#
#     def __init__(self, from_json=None, y_var=False, **fit_params):
#         self.fit_params = fit_params
#         self.from_json = from_json
#         self.y_var = y_var
#         self._initialize()
#
#     def _initialize(self):
#         """Set default values for modifiable parameters."""
#         self._intent_map = {}
#         self._pipeline_map = {}
#         self._choice_map = {}
#         self._multi_column_map = []
#         self._intent_pipelines = {}
#         self._intent_trace = []
#         self.pipeline = None
#         self.is_fit = False
#         self.is_linear = False
#         self._init_json()
#
#     def _get_columns(self, intent):
#         """Iterate columns in intent_map.
#         If intent or intent in its hierarchy matches :param:`intent` then it is
#         appended to a list which is returned. Effectively returns all columns
#         related to an intent.
#         Args:
#             intent: the specific intent class
#         Returns:
#             list(str): A list of the relevant columns to an intent
#         """
#         return [
#             "{}".format(k)
#             for k, v in self._intent_map.items()
#             if intent in inspect.getmro(v)
#         ]
#
#     def _map_intents(self, X_df):
#         """Iterate columns in :param:`X_df`.
#         For each column, the intent tree is traversed and the best-match is
#         returned. This key value pair is added to a dictionary. Any current
#         values in the intent map are allowed to override any new values, this
#         allows  to override the automatic system.
#         Args:
#             X_df (:obj:`DataFrame <pandas.DataFrame>`): input X dataframe
#         """
#         temp_map = {}
#         columns = X_df.columns
#         # Iterate columns
#         for c in columns:
#             if c in self._intent_map:
#                 # column is already mapped to an intent, no need to do the
#                 # traverse here
#                 self._choice_map[c] = [(0, self._intent_map[c])]
#             else:
#                 col_data = X_df.loc[:, [c]]
#                 # Traverse intent tree
#                 valid_cols = [
#                     (i, k)
#                     for i, k in enumerate(GenericIntent.priority_traverse())
#                     if k.is_intent(col_data)
#                 ]
#                 self._choice_map[c] = valid_cols
#                 temp_map[c] = valid_cols[-1][1]
#
#         # Set intent map with override
#         self._intent_map = {**temp_map, **self._intent_map}
#
#     def _build_dependency_order(self):
#         """Create the order in which multi_pipelines need to be executed.
#         They are executed using a "bottom up" technique.
#         """
#         # Dict of intent orders (distance from root node)
#         intent_order = {}
#         # Iterates intents in use
#         for intent in set(self._intent_map.values()):
#
#             # Skip if already in order dict
#             if intent.__name__ in intent_order.keys():
#                 continue
#
#             # Use slice to remove own class and object and BaseIntent
#             # superclass
#             parents = inspect.getmro(intent)[1:-2]
#             # Determine order using class hierarchy
#             order = len(parents)
#
#             # Add to order dict
#             intent_order[intent.__name__] = order
#
#             # Iterate parents
#             for p in parents:
#                 # Skip if parent is already in dict
#                 if p.__name__ in intent_order.keys():
#                     break
#                 # Decrement order (went up a level)
#                 order -= 1
#                 # Add to order dict
#                 intent_order[p.__name__] = order
#
#         # User order dict to sort all active intents
#         intent_list = [(n, i) for i, n in intent_order.items()]
#         intent_list.sort(key=lambda x: x[0])
#         # Evaluate intents into class objects (stored in dict as strings)
#         self._intent_trace = [registry_eval(i) for n, i in intent_list]
#
#     def _map_pipelines(self):
#         """Create single and multi-column pipelines."""
#         # Create single pipeline map
#         self._pipeline_map = {
#             # Creates pipeline object from intent single_pipeline attribute
#             **{
#                 k: SerializablePipeline(
#                     deepcopy(v.single_pipeline(self.y_var))
#                 )
#                 for k, v in self._intent_map.items()
#                 if v.__name__ not in self._intent_pipelines.keys()
#                 and len(v.single_pipeline(self.y_var)) > 0
#             },
#             # Extracts already resolved single pipelines from JSON intent
#             # overrides
#             **{
#                 k: deepcopy(
#                     self._intent_pipelines[v.__name__].get(
#                         "single",
#                         SerializablePipeline(
#                             v.single_pipeline(self.y_var)
#                             if len(v.single_pipeline(self.y_var)) > 0
#                             else [("null", None)]
#                         ),
#                     )
#                 )
#                 for k, v in self._intent_map.items()
#                 if v.__name__ in self._intent_pipelines.keys()
#             },
#             # Column-level pipeline overrides (highest priority)
#             **self._pipeline_map,
#         }
#
#         # Determine order of multi pipeline execution
#         self._build_dependency_order()
#
#         # Build multi pipelines
#         self._intent_pipelines = {
#             # Iterate intents to execute
#             v.__name__: {
#                 # Fetch multi pipeline from Intent class
#                 "multi": SerializablePipeline(
#                     deepcopy(v.multi_pipeline(self.y_var))
#                     if len(v.multi_pipeline(self.y_var)) > 0
#                     else [("null", None)]
#                 ),
#                 "single": SerializablePipeline(
#                     deepcopy(v.single_pipeline(self.y_var))
#                     if len(v.single_pipeline(self.y_var)) > 0
#                     else [("null", None)]
#                 ),
#                 # Extract multi pipeline from JSON config (highest priority)
#                 **{
#                     k: v
#                     for k, v in self._intent_pipelines.get(
#                         v.__name__, {}
#                     ).items()
#                 },
#             }
#             for v in self._intent_trace
#         }
#
#     def _construct_parallel_pipeline(self):
#         """Convert column:pipeline into a ParallelProcessor.
#         Returns:
#             :obj:`ParallelProcessor`
#         """
#         processors = [
#             (col, pipe, [col])
#             for col, pipe in self._pipeline_map.items()
#             if pipe.steps[0][0] != "null"
#         ]
#         if len(processors) == 0:
#             return None
#         return ParallelProcessor(processors)
#
#     def _construct_multi_pipeline(self):
#         """Repack :attr:`_intent_pipeline`.
#         Repacks the dict of list of tuples into Pipeline.
#         Returns:
#             :obj:`Pipeline <sklearn.pipeline.Pipeline>`
#         """
#         # Extract pipelines from postprocess section of JSON config
#         multi_processors = [
#             (val[0], ParallelProcessor([(val[0], val[2], val[1])]))
#             for val in self._multi_column_map
#             if val[2].steps[0][0] != "null"
#         ]
#
#         # Construct multi pipeline from intent trace and intent pipeline
#         # dictionary
#         processors = [
#             (
#                 intent.__name__,
#                 ParallelProcessor(
#                     [
#                         (
#                             intent.__name__,
#                             self._intent_pipelines[intent.__name__]["multi"],
#                             self._get_columns(intent),
#                         )
#                     ]
#                 ),
#             )
#             for intent in reversed(self._intent_trace)
#             if self._intent_pipelines[intent.__name__]["multi"].steps[0][
#                 PipelineStep["NAME"]
#             ]
#             != "null"
#         ]
#
#         if len(multi_processors + processors) == 0:
#             return None
#
#         # Return pipeline with multi_pipeline transformers and postprocess
#         # transformers
#         return SerializablePipeline(processors + multi_processors)
#
#     def _construct_linear_pipeline(self, X_df):
#         """Get single pipeline from pipeline map.
#         Args:
#             X_df (:obj:`DataFrame <pandas.DataFrame>`): input X dataframe
#         Returns:
#             :obj:`Pipeline <sklearn.pipeline.Pipeline>`
#         """
#         return self._pipeline_map.get(
#             X_df.columns[0], SerializablePipeline([("null", None)])
#         )
#
#     def _generate_pipeline(self, X_df):
#         """Construct the final internal pipeline to be used.
#         Args:
#             X_df (:obj:`DataFrame <pandas.DataFrame>`): input X dataframe
#         """
#         # Parse JSON config and populate intent_map and pipeline_map
#         self._initialize()
#
#         # Map intents to columns
#         self._map_intents(X_df)
#
#         # Map pipelines to columns
#         self._map_pipelines()
#
#         # If a single column construct a simple linear pipeline
#         if len(X_df.columns) == 1:
#             self.pipeline = self._construct_linear_pipeline(X_df)
#             self.is_linear = True
#
#         # Else construct a full pipeline
#         else:
#
#             # Per-column single pipelines
#             parallel = self._construct_parallel_pipeline()
#
#             # Multi pipelines
#             multi = self._construct_multi_pipeline()
#
#             # Verify null pipeline isn't added
#             pipe = []
#             if parallel:
#                 pipe.append(("single", parallel))
#             if multi:
#                 pipe.append(("multi", multi))
#
#             # Ensure output of pipeline has a single index
#             pipe.append(
#                 (
#                     "collapse",
#                     ParallelProcessor(
#                         [("null", None, [])], collapse_index=True
#                     ),
#                 )
#             )
#
#             self.pipeline = SerializablePipeline(pipe)
#
#     def _init_json(self):
#         """Load and parse JSON config.
#         Raises:
#             ValueError: Invalid key passed into configuration dictionary.
#             val_err: Pass through ValueErrors when there are issues with
#                 pipeline resolution
#         """
#         config = self.from_json
#         if config is None:
#             return
#
#         try:
#             if "y_var" in config.keys():
#                 self.y_var = config["y_var"]
#             # Parse columns section
#             if "columns" in config.keys():
#                 # Iterate columns
#                 for k, v in config["columns"].items():
#                     # Assign custom intent map
#                     self._intent_map[k] = registry_eval(v["intent"])
#
#                     # Assign custom pipeline map
#                     if "pipeline" in v.keys():
#                         self._pipeline_map[k] = _resolve_pipeline(
#                             v["pipeline"]
#                         )
#
#             # Resolve postprocess section into a list of pipelines
#             if "postprocess" in config.keys():
#                 self._multi_column_map = [
#                     [v["name"], v["columns"], _resolve_pipeline(v["pipeline"])]
#                     for v in config["postprocess"]
#                     if _validate_pipeline(v)
#                 ]
#
#             # Resolve intents section into a dictionary of intents and
#             # pipelines
#             if "intents" in config.keys():
#                 self._intent_pipelines = {
#                     k: {l: _resolve_pipeline(j) for l, j in v.items()}
#                     for k, v in config["intents"].items()
#                 }
#         except KeyError as e:
#             raise ValueError(
#                 "JSON Configuration is malformed: {}".format(str(e))
#             )
#         except ValueError as val_err:
#             raise val_err
#
#     def get_params(self, deep=True):
#         """Get parameters for this estimator.
#         Args:
#             deep (bool): If True, will return the parameters for this estimator
#                 and contained subobjects that are estimators.
#         Returns:
#             dict: returns a dictionary estimator parameters
#         """
#         if self.pipeline is None:
#             return {"from_json": self.from_json}
#         return {
#             "from_json": self.from_json,
#             **self.pipeline.get_params(deep=deep),
#         }
#
#     def set_params(self, **params):
#         """Set the parameters of this estimator.
#         Args:
#             **params (dict): Valid parameter keys can be listed with
#                 :meth:`get_params()`.
#         """
#         self.from_json = params.pop("from_json", self.from_json)
#         self._init_json()
#
#         if self.pipeline is None:
#             return
#
#         self.pipeline.set_params(**params)
#
#     def serialize(self):
#         """Serialize internal arguments and logic.
#         Creates a python dictionary that represents the processes used to
#         transform the dataframe. This can be exported to a JSON file and
#         modified in order to change pipeline behavior.
#         Returns:
#             dict: (See user guide for more detail)
#         """
#         json_cols = {
#             k: {
#                 "intent": self._intent_map[k].__name__,
#                 "pipeline": _serialize_pipeline(
#                     self._pipeline_map.get(
#                         k, SerializablePipeline([("null", None)])
#                     )
#                 ),
#                 "all_matched_intents": [
#                     c[1].__name__ for c in self._choice_map[k]
#                 ],
#             }
#             for k in self._intent_map.keys()
#         }
#
#         # Serialize multi-column processors
#         json_multi = [
#             {
#                 "name": v[0],
#                 "columns": v[1],
#                 "pipeline": _serialize_pipeline(v[2]),
#             }
#             for v in self._multi_column_map
#         ]
#
#         # Serialize intent multi processors
#         json_intents = {
#             k: {
#                 l: _serialize_pipeline(j, include_smart=True)
#                 for l, j in v.items()
#             }
#             for k, v in self._intent_pipelines.items()
#         }
#
#         return {
#             "columns": json_cols,
#             "postprocess": json_multi,
#             "intents": json_intents,
#             "y_var": self.y_var,
#         }
#
#     def summarize(self, X_df):
#         """Generate statistics for each column.
#         Args:
#             X_df (:obj:`DataFrame <pandas.DataFrame>`): input X dataframe
#         Returns:
#             dict: A json dictionary of values with each key representing a \
#                 column and its the value representing the results of that \
#                 intent's :meth:`column_summary()` function
#         """
#         return {
#             k: {
#                 "intent": self._intent_map[k].__name__,
#                 "data": self._intent_map[k].column_summary(X_df[[k]]),
#             }
#             for k in self._intent_map.keys()
#         }
#
#     def fit(self, X, y=None, **fit_params):
#         """Fit internal pipeline to X data.
#         Args:
#             X (:obj:`DataFrame <pandas.DataFrame>`): Input data to be
#                 transformed
#             y (:obj:`DataFrame <pandas.DataFrame>`): Response data
#             **fit_params: Additional fit parameters
#         Returns:
#             :obj:`Pipeline <sklearn.pipeline.Pipeline>`: Fitted internal
#                 pipeline
#         """
#         X = check_df(X)
#         y = check_df(y, ignore_none=True)
#         self.from_json = fit_params.pop("from_json", self.from_json)
#         self._generate_pipeline(X)
#         # import pdb
#         # pdb.set_trace()
#         self.is_fit = True
#         return self.pipeline.fit(X, y, **fit_params)
#
#     def transform(self, X):
#         """Transform X using internal pipeline.
#         Args:
#             X (:obj:`DataFrame <pandas.DataFrame>`): Input data to be
#                 transformed
#         Returns:
#             :obj:`DataFrame <pandas.DataFrame>`: DataFrame of transformed data
#         Raises:
#             ValueError: If pipeline is not fit
#         """
#         X = check_df(X)
#         if not self.pipeline:
#             raise ValueError("Pipeline not fit!")
#         return self.pipeline.transform(X)
#
#     def inverse_transform(self, X):
#         """Invert transform on X using internal pipeline.
#         Args:
#             X (:obj:`DataFrame <pandas.DataFrame>`): Input data to be
#                 transformed
#         Returns:
#             :obj:`DataFrame <pandas.DataFrame>`: DataFrame of inverse
#                 transformed data
#         Raises:
#             ValueError: If the pipeline isn't fit or cannot transform
#             ValueError: If pipeline doesn't support inverse transforms
#         """
#         X = check_df(X)
#         if not self.pipeline or not self.is_fit:
#             raise ValueError("Pipeline not fit, cannot transform.")
#         if not self.is_linear:
#             raise ValueError("Pipeline does not support inverse transform!")
#         return self.pipeline.inverse_transform(X)
#
#
# def _ser_params(trans):
#     """Serialize transformer parameters.
#     Args:
#         trans: a specific sklearn compatible transformer that extends
#             :obj:`BaseEstimator <sklearn.base.BaseEstimator>`
#     Returns:
#         dict: the transformer's parameters
#     """
#     from foreshadow.transformers.concrete import no_serialize_params
#
#     bad_params = ["name", *no_serialize_params.get(type(trans).__name__, [])]
#     return {
#         k: v
#         for k, v in trans.get_params(deep=False).items()
#         if k not in bad_params
#     }
#
#
# def _serialize_pipeline(pipeline, include_smart=False):
#     """Serialize :obj:`Pipeline <sklearn.pipeline.Pipeline>`.
#     Serializes object into JSON object for reconstruction.
#     Args:
#         pipeline (:obj:`Pipeline <sklearn.pipeline.Pipeline>`): Pipeline object
#             to serialize
#         include_smart (bool, optional): whether or not to include smart
#             transformers or to resolve to concrete transformers
#     Returns:
#         list(dict): JSON serializable object of form \
#             :code:`[cls, name, {**params}]`
#     """
#     return [
#         p
#         for s in [
#             (
#                 [
#                     {
#                         "transformer": type(
#                             step[PipelineStep["CLASS"]].transformer
#                         ).__name__,
#                         "name": step[PipelineStep["NAME"]],
#                         "parameters": _ser_params(
#                             step[PipelineStep["CLASS"]].transformer
#                         ),
#                     }
#                 ]
#                 if not isinstance(
#                     step[PipelineStep["CLASS"]].transformer,
#                     SerializablePipeline,
#                 )
#                 else _serialize_pipeline(
#                     step[PipelineStep["CLASS"]].transformer
#                 )
#             )
#             if isinstance(step[PipelineStep["CLASS"]], SmartTransformer)
#             and not include_smart
#             else [
#                 {
#                     "transformer": type(step[PipelineStep["CLASS"]]).__name__,
#                     "name": step[PipelineStep["NAME"]],
#                     "parameters": _ser_params(step[PipelineStep["CLASS"]]),
#                 }
#             ]
#             for step in pipeline.steps
#             if pipeline.steps[0][PipelineStep["NAME"]] != "null"
#         ]
#         for p in s
#     ]
#
#
# def _resolve_pipeline(pipeline_json):
#     """Deserializes pipeline from JSON into sklearn Pipeline object.
#     Args:
#         pipeline_json: list of form :code:`[cls, name, {**params}]`
#     Returns:
#         :obj:`Pipeline <sklearn.pipeline.Pipeline>`: Pipeline based on JSON
#     Raises:
#         KeyError: Malformed transformer while deserializing pipeline
#         ValueError: Cannot import a defined transformer
#         ValueError: Invalid parameters passed into transformer
#     """
#     pipe = []
#
#     for trans in pipeline_json:
#
#         try:
#             clsname = trans["transformer"]
#             name = trans["name"]
#             params = trans.get("parameters", {})
#
#         except KeyError:
#             raise KeyError(
#                 "Malformed transformer {} correct syntax is"
#                 '["transformer": cls, "name": name, "pipeline": '
#                 "{{**params}}]".format(trans)
#             )
#
#         try:
#             cls = get_transformer(clsname)
#
#         except Exception:
#             raise ValueError(
#                 "Could not import defined transformer {}".format(clsname)
#             )
#
#         try:
#             pipe.append((name, cls(**params)))
#         except TypeError:
#             raise ValueError(
#                 "Params {} invalid for transfomer {}".format(
#                     params, cls.__name__
#                 )
#             )
#
#     if len(pipe) == 0:
#         return SerializablePipeline([("null", None)])
#
#     return SerializablePipeline(pipe)
#
#
# def _validate_pipeline(v):
#     """Validate a pipeline dictionary.
#     Check if it contains the correct keys for a pipeline.
#     Args:
#         v (dict): Pipeline dictionary
#     Returns:
#         True if dict is valid pipeline
#     """
#     return (
#         "columns" in v.keys() and "pipeline" in v.keys() and "name" in v.keys()
#     )
