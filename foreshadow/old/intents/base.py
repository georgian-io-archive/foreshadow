"""Intent base and registry definitions."""
# flake8: noqa
#
# from abc import abstractmethod
# from collections import namedtuple
# from functools import wraps
#
# from foreshadow.intents.registry import _IntentRegistry, registry_eval
#
#
# # must be defined above registry import
# PipelineTemplateEntry = namedtuple(
#     "PipelineTemplateEntry", ["transformer_name", "transformer_entry", "y_var"]
# )
#
#
# TransformerEntry = namedtuple("TransformerEntry", ["transformer", "args_dict"])
#
#
# def check_base(ofunc):  # noqa: D202
#     """Get classmethods decorator to check if called from BaseIntent.
#     .. # noqa: I101
#     .. # noqa: I201
#     .. # noqa: I401
#     """
#
#     @wraps(ofunc)
#     def nfunc(*args, **kwargs):
#         if args[0].__name__ == "BaseIntent":
#             raise TypeError(
#                 "classmethod {} cannot be called on BaseIntent".format(
#                     ofunc.__name__
#                 )
#             )
#         return ofunc(*args, **kwargs)
#
#     return nfunc
#
#
# class BaseIntent(metaclass=_IntentRegistry):
#     """Base Class for defining the concept of an "Intent" within Foreshadow.
#     Provides the base infrastructure for the fundamental functionality of an
#     intent. An intent is a singleton class that serves as the fundamental
#     logical component of Foreshadow and exists as part of a ordered tree
#     hierarchy.
#     Intents contain a single pipeline and a multi pipeline attribute. These
#     attributes determine the operations that are applied to features to which
#     that intent is assigned. Single pipelines are individually applied to
#     single columns and multi pipelines are applied across all columns of a
#     given intent.
#     When a column is encountered in a data frame, all intents loaded into the
#     Foreshadow system are iterated in a top-down fashion from least-specific
#     intent to most-specific intent until an intent is found for which no more
#     specific intents match.
#     To support this process each Intent contains a is_intent classmethod which
#     is independently executed to determine whether a specific intent (and its
#     pipelines) should be applied to a feature. The most specific intent for
#     which is_intent returns true is the intent that is assigned to that
#     feature.
#     Attributes:
#         single_pipeline: Pipeline that is executed on a single column. Can
#             create multiple columns.
#         multi_pipeline: Pipeline that is executed across all columns of the
#             given intent. Can reduce or create more columns.
#     """
#
#     children = None
#     """More-specific intents that require this intent to match to be
#     considered.
#     """
#
#     single_pipeline_template = None
#     """A template for single pipelines of smart transformers that affect a
#     single column in an intent. Uses a list of PipelineTemplateEntry to
#     describe the transformers.
#     The template needs an additional boolean at the end of the constructor
#     that determines whether the transformation can be applied to response
#     variables::
#         single_pipeline_template = [
#             PipelineTemplateEntry('t1', Transformer1, False),
#             PipelineTemplateEntry('t2', (Transformer2, {'arg1': True}), True),
#             PipelineTemplateEntry('t3', Transformer1, True),
#         ]
#     """
#
#     multi_pipeline_template = None
#     """A template for multi pipelines of smart transformers that affect
#     multiple columns in an intent
#     See single_pipeline_template for an example definition
#     """
#
#     @classmethod
#     @check_base
#     def to_string(cls, level=0):
#         """Get string representation of intent.
#         Intended to assist in visualizing tree.
#         Args:
#             cls(:class:`BaseIntent  <foreshadow.intents.base.BaseIntent>`):
#                 Root node of intent tree to visualize
#         Returns:
#             str: ASCII Intent Tree visualization
#         .. # noqa: S001
#         """
#         ret = "\t" * level + str(cls.__name__) + "\n"
#
#         # Recursive evaluation of class children to create tree
#         for c in cls.children:
#             klass = registry_eval(c)
#             temp = klass.to_string(level + 1)
#             ret += temp
#         return ret
#
#     @classmethod
#     @check_base
#     def priority_traverse(cls):
#         """Traverse intent tree downward from Intent.
#         Args:
#             cls(:class:`BaseIntent  <foreshadow.intents.base.BaseIntent>`):
#                 Class of intent to start traversal from
#         .. # noqa: S001
#         """
#         lqueue = [cls]
#         # Returns intent and evaluates children. Adds children to list to be
#         # evaluated. Results in top-down search of tree
#         while len(lqueue) > 0:
#             yield lqueue[0]
#             node = lqueue.pop(0)
#             if len(node.children) > 0:
#                 node_children = map(registry_eval, node.children)
#                 lqueue[
#                     0:0
#                 ] = node_children  # Append to beginning to do depth first
#
#     @classmethod
#     @check_base
#     @abstractmethod
#     def is_intent(cls, df):
#         """Determine whether intent is the appropriate fit.
#         Args:
#             df: pd.DataFrame to determine intent fit
#         Returns:
#             bool: determines whether intent is valid for feature in df
#         .. # noqa: I202
#         """
#         pass  # pragma: no cover
#
#     @classmethod
#     @check_base
#     @abstractmethod
#     def column_summary(cls, df):
#         """Compute relevant statistics and returns a JSON dict of those values.
#         Args:
#             df: pd.DataFrame to summarize
#         Returns:
#             dict: A JSON representation of relevant statistics
#         .. # noqa: I202
#         """
#         pass  # pragma: no cover
#
#     @classmethod
#     def _check_intent(cls):
#         """Validate class variables are setup properly.
#         .. # noqa: I401
#         """
#         define_attrs = [
#             "children",
#             "single_pipeline_template",
#             "multi_pipeline_template",
#         ]
#         # Check that intent attrs are defined
#         for a in define_attrs:
#             if getattr(cls, a) is None:
#                 raise NotImplementedError(
#                     (
#                         "Subclass must define {} class attribute.\n"
#                         "Developers please see the documentation."
#                     ).format(a)
#                 )
