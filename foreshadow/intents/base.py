"""
Intent base and registry defentions
"""

from abc import abstractmethod
from collections import namedtuple
from functools import wraps

# must be defined above registry import
PipelineTemplateEntry = namedtuple(
    'PipelineTemplateEntry',
    ['transformer_name', 'transformer_entry', 'y_var']
)


TransformerEntry = namedtuple(
    'TransformerEntry',
    ['transformer', 'args_dict']
)

from .registry import _IntentRegistry, registry_eval


def check_base(ofunc):
    """Decorator to wrap classmethods to check if they are being called from BaseIntent"""

    @wraps(ofunc)
    def nfunc(*args, **kwargs):
        if args[0].__name__ == "BaseIntent":
            raise TypeError(
                "classmethod {} cannot be called on BaseIntent".format(ofunc.__name__)
            )
        return ofunc(*args, **kwargs)

    return nfunc


class BaseIntent(metaclass=_IntentRegistry):
    """Base Class for defining the concept of an "Intent" within Foreshadow.

    Provides the base infrastructure for the fundamental functionality of an intent.
    An intent is a singleton class that serves as the fundamental logical component
    of Foreshadow and exists as part of a ordered tree hierarchy.

    Intents contain a single pipeline and a multi pipeline attribute. These
    attributes determine the operations that are applied to features to which that
    intent is assigned. Single pipelines are individually applied to single columns
    and multi pipelines are applied across all columns of a given intent.

    When a column is encountered in a data frame, all intents loaded into the
    Foreshadow system are iterated in a top-down fashion from least-specific intent
    to most-specific intent until an intent is found for which no more specific
    intents match.

    To support this process each Intent contains a is_intent classmethod which is
    independently executed to determine whether a specific intent (and its pipelines)
    should be applied to a feature. The most specific intent for which is_intent
    returns true is the intent that is assigned to that feature.

    Attributes:
        single_pipeline: Pipeline that is executed on a single column. Can create
            multiple columns.
        multi_pipeline: Pipeline that is executed across all columns of the given
            intent. Can reduce or create more columns.

    """

    children = None
    """More-specific intents that require this intent to match to be
            considered."""

    single_pipeline_template = None
    """A template for single pipelines of smart transformers that affect a 
        single column in an intent

        The template needs an additional boolean at the end of the tuple that
        determines whether the transformation can be applied to response 
        variables.
    
        Example: single_pipeline_template = [
            ('t1', Transformer1, False),
            ('t2', (Transformer2, {'arg1': True}), True),
            ('t3', Transformer1, True),
        ]
    """

    multi_pipeline_template = None
    """A template for multi pipelines of smart transformers that affect multiple 
        columns in an intent
    
        See single_pipeline_template for an example defention
    """

    @classmethod
    @check_base
    def to_string(cls, level=0):
        """String representation of intent. Intended to assist in visualizing tree.

        Args:
            cls(:class:`BaseIntent  <foreshadow.intents.base.BaseIntent>`):  Root node
                of intent tree to visualize

        Returns:
            str: ASCII Intent Tree visualization

        """
        ret = "\t" * level + str(cls.__name__) + "\n"

        # Recursive evaluation of class children to create tree
        for c in cls.children:
            klass = registry_eval(c)
            temp = klass.to_string(level + 1)
            ret += temp
        return ret

    @classmethod
    @check_base
    def priority_traverse(cls):
        """Traverses intent tree downward from Intent.

        Args:
            cls(:class:`BaseIntent  <foreshadow.intents.base.BaseIntent>`): Class of
                intent to start traversal from

        """
        lqueue = [cls]
        # Returns intent and evaluates children. Adds children to list to be evaluated
        # Results in top-down search of tree
        while len(lqueue) > 0:
            yield lqueue[0]
            node = lqueue.pop(0)
            if len(node.children) > 0:
                node_children = map(registry_eval, node.children)
                lqueue.extend(node_children)

    @classmethod
    @check_base
    @abstractmethod
    def is_intent(cls, df):
        """Determines whether intent is the appropriate fit

        Args:
            df: pd.DataFrame to determine intent fit

        Returns:
            Boolean determining whether intent is valid for feature in df
        """
        pass  # pragma: no cover

    @classmethod
    def _check_intent(cls):
        """Validate class variables are setup properly"""
        not_implemented = lambda v, m: "Subclass must define {} class attribute.\n{}".format(
            v, m
        )
        define_attrs = [
            "children",
            "single_pipeline_template",
            "multi_pipeline_template",
        ]
        # Check that intent attrs are defined
        for a in define_attrs:
            if getattr(cls, a) is None:
                raise NotImplementedError(
                    not_implemented(a, "Developers please see the documentation.")
                )
