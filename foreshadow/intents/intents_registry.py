"""
Intent Registry
"""

from collections import Iterable

_registry = {}
_intent_tree = None


def _register_intent(cls_target):
    global _registry, _intent_tree
    if cls_target.__name__ in _registry:
        raise TypeError("Intent already exists in registry, use a different name")
    _registry[cls_target.__name__] = cls_target


def unregister_intent(cls_target):
    global _registry
    if not isinstance(cls_target, str) and all(isinstance(s, str) for s in cls_target):
        for c in cls_target:
            if c in _registry:
                del _registry[c]
    elif isinstance(cls_target, str):
        if cls_target in _registry:
            del _registry[cls_target]


def get_registry():
    global _registry
    return _registry


def registry_eval(cls_target):
    global _registry
    return _registry.get(cls_target, None)


class IntentRegistry(type):
    def __new__(meta, name, bases, class_dict):
        klass = type.__new__(meta, name, bases, class_dict)
        if not name == "BaseIntent":
            klass._check_required_class_attributes()
            _register_intent(klass)
        return klass

    def __str__(self, level=0):
        return self.tostring()
