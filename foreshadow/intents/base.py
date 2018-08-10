"""
Intent base and registry defentions
"""

from .registry import _IntentRegistry, registry_eval


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
        dtype: Data type of column required for this intent to match
        children: More-specific intents that require this intent to match to be
            considered.
        single_pipeline: Pipeline that is executed on a single column. Can create
            multiple columns.
        multi_pipeline: Pipeline that is executed across all columns of the given
            intent. Can reduce or create more columns.

    """

    dtype = None
    children = None

    def __new__(cls, *args, **kwargs):
        if cls is BaseIntent:
            raise TypeError("BaseIntent may not be instantiated")
        return object.__new__(cls, *args, **kwargs)

    @classmethod
    def to_string(cls, level=0):
        """String representation of intent. Intended to assist in visualizing tree."""
        ret = "\t" * level + str(cls.__name__) + "\n"
        for c in cls.children:
            klass = registry_eval(c)
            temp = klass.to_string(level + 1)
            ret += temp
        return ret

    @classmethod
    def priority_traverse(cls):
        """Traverses intent tree downward from Intent."""
        lqueue = [cls]
        while len(lqueue) > 0:
            yield lqueue[0]
            node = lqueue.pop(0)
            if len(node.children) > 0:
                node_children = map(registry_eval, node.children[::-1])
                lqueue.extend(node_children)

    @classmethod
    def is_intent(cls, df):
        """Determines whether intent is the appropriate fit

        Args:
            df: pd.DataFrame to determine intent fit

        Returns:
            Boolean determining whether intent is valid for feature in df
        """
        raise NotImplementedError("is_fit is not immplemented")

    @classmethod
    def _check_required_class_attributes(cls):
        """Validate class variables are setup properly"""
        not_implemented = lambda x, y: "Subclass must define {} attribute.\n{}".format(
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
