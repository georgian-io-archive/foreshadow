"""
Intent base and registry defentions
"""

from .intents_registry import IntentRegistry, registry_eval


class BaseIntent(object, metaclass=IntentRegistry):
    dtype = None
    children = None

    def __new__(cls, *args, **kwargs):
        if cls is BaseIntent:
            raise TypeError("BaseIntent may not be instantiated")
        return object.__new__(cls, *args, **kwargs)

    @classmethod
    def tostring(cls, level=0):
        ret = "\t" * level + str(cls.__name__) + "\n"
        for c in cls.children:
            klass = registry_eval(c)
            temp = klass.tostring(level + 1)
            ret += temp
        return ret

    @classmethod
    def priority_traverse(cls):
        lqueue = [cls]
        while len(lqueue) > 0:
            yield lqueue[0]
            node = lqueue.pop(0)
            node_children = filter(
                lambda x: not x is None, map(registry_eval, reversed(node.children))
            )
            lqueue.extend(node_children)

    @classmethod
    def is_intent(cls, df):
        """Determines whether intent is the appropriate fit

        :param df (pd.DataFrame): data to determine intent fit

        :returns: bool
        """
        raise NotImplementedError("is_fit is not immplemented")

    @classmethod
    def get_best_single_pipeline(cls, df):
        raise NotImplementedError("get_best_single_pipeline is not immplemented")

    @classmethod
    def get_best_multi_pipeline(cls, df):
        raise NotImplementedError("get_best_multi_pipeline is not immplemented")

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
