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

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj._verify_pipeline_setup()
        return obj


class BaseIntent(object, metaclass=IntentRegistry):
    intent = None
    dtype = None
    parent = None
    children = None
    single_pipeline_spec = None
    multi_pipeline_spec = None

    def __new__(cls, *args, **kwargs):
        if cls is BaseIntent:
            raise TypeError("BaseIntent may not be instantiated")
        return object.__new__(cls, *args, **kwargs)

    @classmethod
    def is_fit(cls, series):
        """Determines whether intent is the appropriate fit

        :param series (pd.Series): data to determine intent fit

        :returns: bool
        """
        raise NotImplementedError("Function is not immplemented")

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
        elif cls.parent is None:
            raise NotImplementedError(
                not_implemented(
                    "cls.parent",
                    "This attribute should define the parent of the intent.",
                )
            )
        elif cls.children is None:
            raise NotImplementedError(
                not_implemented(
                    "cls.children",
                    "This attribute should define the children of the intent.",
                )
            )
        elif cls.single_pipeline_spec is None:
            raise NotImplementedError(
                not_implemented(
                    "cls.single_pipeline_spec",
                    (
                        "This attribute should define the sklearn single "
                        "pipeline spec of the intent."
                    ),
                )
            )
        elif cls.multi_pipeline_spec is None:
            raise NotImplementedError(
                not_implemented(
                    "cls.multi_pipeline_spec",
                    (
                        "This attribute should define the sklearn multi pipeline "
                        "spec of the intent."
                    ),
                )
            )

    def _verify_pipeline_setup(self):
        not_implemented = lambda x, y: "Subclass initialize {} to {}".format(x, y)
        if (
            not hasattr(self, "single_pipeline")
            or self.single_pipeline_spec != self.single_pipeline
        ):
            raise NotImplementedError(
                not_implemented("self.single_pipeline", "self.single_pipeline_spec")
            )
        elif not hasattr(self, "multi_pipeline"):
            raise NotImplementedError(
                not_implemented("self.multi_pipeline", "self.multi_pipeline_spec")
            )
