"""Intent Registry."""
# flake8: noqa
# from abc import ABCMeta
#
# from foreshadow.base import BaseEstimator, TransformerMixin
#
# from foreshadow.intents import base
# from foreshadow.core import SmartTransformer
#
#
# _registry = {}
#
#
# def _register_intent(cls_target):
#     """Register an intent with foreshadow.
#     Args:
#         cls_target: intent class target
#     Raises:
#           TypeError: intent already registered
#     """
#     global _registry
#     if cls_target.__name__ in _registry:
#         raise TypeError(
#             "Intent already exists in registry, use a different name"
#         )
#     _registry[cls_target.__name__] = cls_target
#
#
# def _unregister_intent(cls_target):
#     """Remove intent from registry.
#     Args:
#         cls_target: intent class target
#     Raises:
#         ValueError: intent not found
#     """
#     global _registry
#
#     def validate_input(clsname):
#         if clsname not in _registry:
#             raise ValueError("{} was not found in registry".format(clsname))
#
#     if isinstance(cls_target, list) and all(
#         isinstance(s, str) for s in cls_target
#     ):
#         [validate_input(c) for c in cls_target]
#         for c in cls_target:
#             del _registry[c]
#     elif isinstance(cls_target, str):
#         validate_input(cls_target)
#         del _registry[cls_target]
#     else:
#         raise ValueError("Input must be either a string or a list of strings")
#
#
# def _process_templates(cls_target):  # noqa: D202
#     """Process template.
#     Args:
#         cls_target: intent class target
#     Raises:
#         ValueError: error encountered
#     """
#
#     def _resolve_template(template):
#         if not all(
#             isinstance(s, base.PipelineTemplateEntry)
#             and (
#                 (
#                     isinstance(s.transformer_entry, type)
#                     and issubclass(
#                         s.transformer_entry, (BaseEstimator, TransformerMixin)
#                     )
#                 )
#                 or (
#                     isinstance(s.transformer_entry, base.TransformerEntry)
#                     and isinstance(s.transformer_entry.transformer, type)
#                     and issubclass(
#                         s.transformer_entry.transformer,
#                         (BaseEstimator, TransformerMixin),
#                     )
#                     and isinstance(s.transformer_entry.args_dict, dict)
#                 )
#             )
#             for s in template
#         ):
#             raise ValueError("Malformed transformer entry in template")
#
#         x_pipeline = [
#             (
#                 s.transformer_name,
#                 s.transformer_entry()
#                 if callable(s.transformer_entry)
#                 else s.transformer_entry.transformer(
#                     **s.transformer_entry.args_dict
#                 ),
#             )
#             for s in template
#         ]
#         y_pipeline = [
#             (
#                 s.transformer_name,
#                 s.transformer_entry(
#                     **{
#                         "y_var": True
#                         for _ in range(1)
#                         if issubclass(s.transformer_entry, SmartTransformer)
#                     }
#                 )
#                 if callable(s.transformer_entry)
#                 else s.transformer_entry.transformer(
#                     **s.transformer_entry.args_dict,
#                     **{
#                         "y_var": True
#                         for _ in range(1)
#                         if issubclass(
#                             s.transformer_entry.transformer, SmartTransformer
#                         )
#                     }
#                 ),
#             )
#             for s in template
#             if s.y_var
#         ]
#
#         return x_pipeline, y_pipeline
#
#     def _process_template(cls_target, template_name):
#         """Process template.
#         Args:
#             cls_target: intent class target
#             template_name: name of template
#         Returns:
#             template
#         """
#         t = getattr(cls_target, template_name)
#         attr_base = template_name.replace("_template", "")
#         if len(t) == 0:
#             setattr(cls_target, attr_base + "_x", t)
#             setattr(cls_target, attr_base + "_y", t)
#         else:
#             x_pipe, y_pipe = _resolve_template(t)
#             setattr(cls_target, attr_base + "_x", x_pipe)
#             setattr(cls_target, attr_base + "_y", y_pipe)
#
#         return lambda y_var=False: (
#             getattr(cls_target, attr_base + "_x")
#             if not y_var
#             else getattr(cls_target, attr_base + "_y")
#         )
#
#     cls_target.single_pipeline = _process_template(
#         cls_target, "single_pipeline_template"
#     )
#     cls_target.multi_pipeline = _process_template(
#         cls_target, "multi_pipeline_template"
#     )
#
#
# def registry_eval(cls_target):
#     """Retrieve intent class from registry dictionary.
#     Args:
#         cls_target(str): String name of Intent
#     Returns:
#         :class:`BaseIntent <foreshadow.intents.base.BaseIntent>`: Intent class
#             object
#     """
#     return _registry[cls_target]
#
#
# class _IntentRegistry(ABCMeta):
#     """Register defined intent classes using metaclass."""
#
#     def __new__(cls, *args, **kwargs):
#         class_ = super(_IntentRegistry, cls).__new__(cls, *args, **kwargs)
#
#         if class_.__abstractmethods__ and class_.__name__ != "BaseIntent":
#             raise NotImplementedError(
#                 "{} has not implemented abstract methods {}".format(
#                     class_.__name__, ", ".join(class_.__abstractmethods__)
#                 )
#             )
#         elif class_.__name__ != "BaseIntent":
#             class_._check_intent()
#             _process_templates(class_)
#             _register_intent(class_)
#         return class_
