"""Parameter mapping utilities."""
# flake8: noqa
# import itertools
# from copy import deepcopy
#
# # TODO: Write default parameters here
#
# config_dict = {"StandardScaler.with_std": [True, False]}
#
#
# def param_mapping(pipeline, X_df, y_df):
#     """Generate parameter search space.
#
#     Generated using an unfit pipeline and sample X and Y data. This pulls
#     search space information from :code:`config_dict` and from the JSON
#     configuration of the Preprocessor objects in the given Pipeline.
#
#     Args:
#         pipeline (:obj:`Pipeline <sklearn.pipeline.Pipeline>`): Input unfit
#             pipeline
#         X_df: (:obj:`pandas.DataFrame`): Input X dataframe
#         y_df: (:obj:`pandas.DataFrame`): Input y dataframe
#
#     Returns:
#         list: List of dict for which keys are parameters and the value is a \
#             list representing the search space
#
#     """
#     # Get preprocessors from the Pipeline
#     preprocessors = [
#         k
#         for k, v in pipeline.get_params().items()
#         if isinstance(v, Preprocessor)
#     ]
#
#     # For each preprocessor object extract the search space defined in the
#     # config. This dict is of the form preprocessor: list(configs)
#     configs = {
#         p: _parse_json_params(pipeline.get_params()[p].from_json)
#         for p in preprocessors
#     }
#
#     # For each preprocessor zip the list of configs into a list of tuples
#     # of the form (preprocessor, config) and concatenate those lists.
#     #
#     # Apply a inner product to those tuples to get every combination of
#     # configuration. Convert to list of tasks
#     tasks = [
#         {k: tsk for k, tsk in i}
#         for i in itertools.product(
#             *[[(p, t) for t in cfgs] for p, cfgs in configs.items()]
#         )
#     ]
#
#     params = []
#     # Iterate tasks
#     for task in tasks:
#
#         # Set parameters of pipeline using config
#         pipeline.set_params(
#             **{"{}__from_json".format(k): v for k, v in task.items()}
#         )
#
#         # Fit the param on the sample data
#         pipeline.fit(X_df, y_df)
#
#         # Get the full configuration
#         param = pipeline.get_params()
#
#         # Create a dictionary of parameters
#         # One parameter is from_json (the configuration dict we calculated)
#         # Other parameters are pulled from config_dict using
#         # extract_config_params()
#
#         explicit_params = {
#             "{}__from_json".format(k): [param[k].serialize()]
#             for k, v in task.items()
#         }
#
#         params.append({**explicit_params, **_extract_config_params(param)})
#
#     return params
#
#
# def _parse_json_params(from_json):
#     """Generate list of possible configuration files using JSON config.
#
#     Args:
#         from_json (dict): JSON configuration file with combinations section
#
#     Returns:
#         list: List of dictionaries of possible configurations for this \
#             preprocessor
#
#     """
#     if from_json is None:
#         return [None]
#
#     combinations = from_json.pop("combinations", [])
#
#     out = []
#     # Iterate combination section of config
#     for combo in combinations:
#
#         # For each parameter to search, split into list of tuples
#         t = [[(k, value) for value in eval(v)] for k, v in combo.items()]
#
#         # Perform product of all lists of tuples
#         c = itertools.product(*t)
#
#         # Expand into list of dictionaries of individual parameter configs
#         d = [{k: v for k, v in i} for i in c]
#
#         # For each dict of parameter configurations create the full
#         # configuration dict needed for the preprocessor using the other
#         # sections of from_json
#         out += [_override_dict(i, from_json) for i in d]
#
#     if len(out) < 1:
#         out.append(None)
#
#     return out
#
#
# def _override_dict(override, original):
#     """Override dictionary with keys from another dict.
#
#     Args:
#         override (dict): Dictionary with override keys
#         original (dict): Dictionary to be overriden
#
#     Returns:
#         dict: Final overridden dictionary
#
#     """
#     temp = deepcopy(original)
#     # Iterate keys and perform overrides
#     for k, v in override.items():
#         _set_path(k, v, temp)
#     return temp
#
#
# def _set_path(key, value, original):
#     """Set the path of a dictionary using a string key.
#
#     Args:
#         key (str): Path in dictionary
#         value: The value to be set
#         original (dict): The dict for which the value is set
#
#     Raises:
#         ValueError: Raises when when given an invalid key path
#
#     """
#     path = key.split(".")
#     temp = original
#     curr_key = ""
#
#     try:
#
#         # Searches in path, indexed by key if dictionary or by index if list
#         for p in path[:-1]:
#             curr_key = p
#             if isinstance(temp, list):
#                 temp = temp[int(p)]
#             else:  # Dictionary
#                 temp = temp[p]
#
#         # Always Dictionary
#         temp[path[-1]] = value
#         if path[-1] == "intent":
#             try:
#                 del temp["pipeline"]
#             except Exception:
#                 pass
#         if path[-1] == "transformer":
#             del temp["parameters"]
#
#     except KeyError:
#         raise ValueError("Invalid JSON Key {} in {}".format(curr_key, temp))
#
#     except ValueError:
#         raise ValueError(
#             "Attempted to index list {} with value {}".format(temp, curr_key)
#         )
#
#
# def _extract_config_params(param):
#     """Extract the configuration parameters from get_params() from a Pipeline.
#
#     Args:
#         param (dict): Result of pipeline.get_params()
#
#     Returns:
#         dict: Dict of parameters for which the key is a list of values to \
#             search
#
#     """
#     out = {}
#
#     # Iterate parameters
#     for k, v in param.items():
#         trace = k.split("__")
#         names = []
#         # Iterate each level in a single parameter
#         for i, t in enumerate(trace[0:-1]):
#             # Append class name and atrribute to names
#             key = "__".join(trace[0 : i + 1])  # noqa: E203
#             name = type(param.get(key, None)).__name__
#             if name not in [
#                 "ParallelProcessor",
#                 "SerializablePipeline",
#                 "Preprocessor",
#                 "FeatureUnion",
#             ]:
#                 names.append(name)
#         names.append(trace[-1])
#         p = ".".join(names)
#         # Check if key is in config_dict and return the search space
#         if p in config_dict:
#             out[k] = list(config_dict[p])
#
#     return out
