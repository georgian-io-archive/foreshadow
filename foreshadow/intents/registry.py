"""
Intent Registry
"""

from abc import ABCMeta

_registry = {}


def _register_intent(cls_target):
    """Registers an intent with the library"""
    global _registry
    if cls_target.__name__ in _registry:
        raise TypeError("Intent already exists in registry, use a different name")
    _registry[cls_target.__name__] = cls_target


def _unregister_intent(cls_target):
    """Removes intent from registry"""
    global _registry

    def validate_input(clsname):
        if clsname not in _registry:
            raise ValueError("{} was not found in registry".format(clsname))

    if isinstance(cls_target, list) and all(isinstance(s, str) for s in cls_target):
        [validate_input(c) for c in cls_target]
        for c in cls_target:
            del _registry[c]
    elif isinstance(cls_target, str):
        validate_input(cls_target)
        del _registry[cls_target]
    else:
        raise ValueError("Input must be either a string or a list of strings")


def registry_eval(cls_target):
    """Retrieve intent class from registry dictionary

    Args:
        cls_target(str): String name of Intent

    Return:
        :class:`BaseIntent <foreshadow.intents.base.BaseIntent>`: Intent class object

    """
    return _registry[cls_target]


class _IntentRegistry(ABCMeta):
    """Metaclass for intents that registers defined intent classes"""

    def __new__(cls, *args, **kwargs):
        class_ = super(_IntentRegistry, cls).__new__(cls, *args, **kwargs)

        if class_.__abstractmethods__ and class_.__name__ is not "BaseIntent":
            raise NotImplementedError(
                "{} has not implemented abstract methods {}".format(
                    class_.__name__, ", ".join(class_.__abstractmethods__)
                )
            )
        elif class_.__name__ is not "BaseIntent":
            class_._check_required_class_attributes()
            _register_intent(class_)
        return class_
