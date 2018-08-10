"""
Intent Registry
"""

_registry = {}


def _register_intent(cls_target):
    """Uses the trained estimator to predict the response for an input dataset

    Args:
        data_df (pandas.DataFrame or numpy.ndarray): The test input data

    Returns:
        pandas.DataFrame: The response variable output (transformed if necessary)
    """
    global _registry
    if cls_target.__name__ in _registry:
        raise TypeError("Intent already exists in registry, use a different name")
    _registry[cls_target.__name__] = cls_target


def _unregister_intent(cls_target):
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


def _set_registry(val):
    global _registry
    _registry = val


def get_registry():
    """Global registry of defined intents"""
    global _registry
    return _registry


def registry_eval(cls_target):
    """Retrieve intent class from registry dictionary"""
    global _registry
    return _registry[cls_target]


class _IntentRegistry(type):
    """Metaclass for intents that registers defined intent classes"""

    def __new__(meta, name, bases, class_dict):
        klass = type.__new__(meta, name, bases, class_dict)
        if not name == "BaseIntent":
            klass._check_required_class_attributes()
            _register_intent(klass)
        return klass
