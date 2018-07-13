"""
Intent base and registry defentions
"""

_registry = {}


def _register_intent(cls_target):
    global _registry
    if cls_target.__name__ in _registry:
        raise TypeError("Intent already exists in registry, use a different name")

    _registry[cls_target.__name__] = cls_target


def unregister_intent(cls_target_str):
    global _registry
    if cls_target_str in _registry:
        del _registry[cls_target_str]


def get_registry():
    global _registry
    return _registry


class IntentRegistry(type):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        if not name == "BaseIntent":
            cls._check_required_class_attributes()
            _register_intent(cls)
        return cls


class BaseIntent(object, metaclass=IntentRegistry):
    intent = None
    dtype = None
    children = None

    def __new__(cls, *args, **kwargs):
        if cls is BaseIntent:
            raise TypeError("BaseIntent may not be instantiated")
        return object.__new__(cls, *args, **kwargs)

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
        if cls.intent is None:
            raise NotImplementedError(
                not_implemented(
                    "cls.intent", "This attribute should define the name of the intent."
                )
            )
        elif cls.dtype is None:
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
