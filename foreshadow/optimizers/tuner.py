"""Classes for optimizing Foreshadow given a param_distribution."""
#
# TODO temporarily comment out since we are not using it and it's dragging
#  down code coverage.
#
# import importlib
# import inspect
#
# from hyperopt import hp
# from sklearn.exceptions import NotFittedError
# from sklearn.utils.validation import check_is_fitted
#
# from foreshadow.base import BaseEstimator, TransformerMixin
#
#
# """
# combinations:
#     X_preparer.cleaner.CHAS:
#         Cleaner:
#             - date:
#                 - p1
#                 - p2
#             - financial
#         IntentMapper:
#             - Something
#
#     X_preparer.cleaner.CHAS.CleanerMapper:
#         -Something
#
#     X_preparer.cleaner.CHAS.IntentMapper:
#         -Something
#
#
#     X_preparer:
#         cleaner:
#             CHAS:
#                 Cleaner:
#                     date:
#                         -p1
#                         -p2
#
#
# Convention:
#     Column name is last. If a .<blank> is present, then applied across all
#     columns.
#
# Things that may be swapped:
#     PreparerSteps,
#     StepSmartTransformers/ConcreteTransformers.
#
# """
#
#
# def _replace_list(key, obj, replace_with=hp.choice):
#     """Recursively replace a nested object's lists with a sampling function.
#
#     Replaces lists/tuples with replace_with.
#
#     Args:
#         key: Current key. Derived from dict keys in nested calls, but should
#             be passed if your top level is a list.
#         obj: the object to have list/tuples replaced.
#         replace_with: Function that takes a key and list and builds a
#             sampling function with it. Built around hp.choice but should be
#             extendable.
#
#     Returns:
#         obj with lists/tuples replaced with replace_with.
#
#     """
#     key = str(key)
#     if isinstance(obj, (tuple, list)):
#         if not isinstance(obj[0], dict):
#             #  we have reached a leaf of parameter specifications.
#             return replace_with(key, obj)
#         else:  # not a leaf, recurse and replace the output.
#             to_replace = []
#             for v in obj:
#                 to_replace.append(_replace_list(key, v, replace_with))
#             return replace_with(key, to_replace)
#     if isinstance(obj, dict):  # not a leaf for sure, we iterate over dict.
#         to_replace = {}
#         for key, v in obj.items():
#             to_replace[key] = _replace_list(key, v, replace_with)
#         obj.update(to_replace)
#         return obj
#     else:  # no nesting and no need to replace.
#         return obj
#
#
# def get(optimizer, **optimizer_kwargs):
#     """Get optimizer from foreshadow.optimizers package.
#
#     Args:
#         optimizer: optimizer name or class
#         **optimizer_kwargs: kwargs used in instantiation.
#
#     Returns:
#         Corresponding instantiated optimizer using kwargs.
#
#     """
#     if isinstance(optimizer, str):
#         mod = importlib.import_module("foreshadow.optimizers")
#         return getattr(mod, optimizer)(**optimizer_kwargs)
#     elif inspect.isclass(optimizer):
#         return optimizer(**optimizer_kwargs)
#     return optimizer
#
#
# class Tuner(BaseEstimator, TransformerMixin):
#     """Tunes the Foreshadow object using a ParamSpec and Optimizer."""
#
#     def __init__(
#         self, pipeline=None, params=None, optimizer=None, optimizer_kwargs={}
#     ):
#         if pipeline is None:
#             raise ValueError(
#                 "'pipeline' is a required arg and is only set to "
#                 "None due to sklearn get_params requirements."
#             )
#         if params is None:
#             raise ValueError(
#                 "'params' is a required arg and is only set to "
#                 "None due to sklearn get_params requirements."
#             )
#         self.pipeline = pipeline
#         self.params = params
#         self.optimizer_kwargs = optimizer_kwargs
#         self.optimizer = get(
#             optimizer,
#             estimator=self.pipeline,
#             param_distributions=self.params,
#             **self.optimizer_kwargs
#         )
#
#     def _reset(self):
#         try:
#             check_is_fitted(self, "best_pipeline")
#             del self.best_pipeline
#             del self.best_params
#         except NotFittedError:
#             pass
#
#     def fit(self, X, y, **fit_params):
#         """Optimize self.pipeline using self.optimizer.
#
#         Args:
#             X: input points
#             y: input labels
#             **fit_params: params to optimizer fit method.
#
#         Returns:
#             self
#
#         """
#         self._reset()
#         self.optimizer.fit(X, y, **fit_params)
#         self.best_pipeline = self.optimizer.best_estimator_
#         self.best_params = self.optimizer.best_params_
#         return self
#
#     def transform(self, pipeline):
#         """Transform pipeline using best_pipeline.
#
#         Args:
#             pipeline: input pipeline
#
#         Returns:
#             best_pipeline.
#
#         """
#         check_is_fitted(self, "best_pipeline")
#         return self.best_pipeline
