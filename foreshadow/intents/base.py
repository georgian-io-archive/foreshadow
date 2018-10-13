"""
Intent base and registry defentions
"""

from abc import abstractmethod
from functools import wraps

from .registry import _IntentRegistry, registry_eval


def check_base(ofunc):
    """Decorator to wrap classmethods to check if they are being called from BaseIntent"""
    @wraps(ofunc)
    def nfunc(*args, **kwargs):
        if args[0].__name__ == 'BaseIntent':
            raise TypeError("classmethod {} cannot be called on BaseIntent".format(ofunc.__name__))
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

    dtype = None
    """Data type of column required for this intent to match (not implemented)"""

    children = None
    """More-specific intents that require this intent to match to be
            considered."""

    single_pipeline = None
    """Single pipeline of smart transformers that affect a single column in 
            in an intent"""

    multi_pipeline = None
    """Multi pipeline of smart transformers that affect multiple columns in
            an intent"""

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
                node_children = map(registry_eval, node.children[::-1])
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
        pass # pragma: no cover

    @classmethod
    def _check_required_class_attributes(cls):
        """Validate class variables are setup properly"""

        not_implemented = lambda x, y: "Subclass must define {} class attribute.\n{}".format(
            x, y
        )
        if cls.dtype is None:
            raise NotImplementedError(
                not_implemented(
                    "cls.dtype", "This attribute should define the dtype of the intent."
                )
            )
        elif cls.children is None:
            raise NotImplementedError(
                not_implemented(
                    "cls.children",
                    "This attribute should define the children of the intent.",
                )
            )
        elif cls.single_pipeline is None:
            raise NotImplementedError(
                not_implemented(
                    "cls.single_pipeline",
                    "This attribute should define the transformers for a single pipeline",
                )
            )
        elif cls.multi_pipeline is None:
            raise NotImplementedError(
                not_implemented(
                    "cls.multi_pipeline",
                    "This attribute should define the transformers for a multi pipeline",
                )
            )
