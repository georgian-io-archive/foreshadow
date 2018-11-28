"""
Intent Registry
"""

from abc import ABCMeta
from foreshadow.transformers.base import SmartTransformer
from sklearn.base import BaseEstimator, TransformerMixin

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


def _process_templates(cls_target):
    def _resolve_template(template):
        if not all(len(s) == 3 for s in template) or not template:
            raise ValueError(
                "Malformed template"
            )
        if not all(
            (
                isinstance(s[1], type)
                and issubclass(s[1], (BaseEstimator, TransformerMixin))
            ) or (
                isinstance(s[1], tuple)
                and len(s[1]) == 2
                and isinstance(s[1][0], type)
                and issubclass(s[1][0], (BaseEstimator, TransformerMixin))
                and isinstance(s[1][1], dict)
            )
            for s in template
        ):
            raise ValueError(
                "Malformed transformer entry in template"
            )

        x_pipeline = [
            (
                s[0], 
                s[1]() if callable(s[1]) else s[1][0](**s[1][1])
            )
            for s in template
        ]
        y_pipeline = [
            (
                s[0], 
                s[1](**{
                    'y_var': True 
                    for _ in range(1) if issubclass(s[1], SmartTransformer)
                }) if callable(s[1]) else s[1][0](
                    **s[1][1],
                    **{
                        'y_var': True 
                        for _ in range(1) if issubclass(s[1][0], SmartTransformer)
                    }
                )
            )
            for s in template
            if s[-1]
        ]

        return x_pipeline, y_pipeline

    def _process_template(cls_target, template_name):
        t = getattr(cls_target, template_name)
        attr_base = template_name.replace('_template', '')
        if len(t) == 0:
            setattr(cls_target, attr_base+'_x', t)
            setattr(cls_target, attr_base+'_y', t)
        else:
            x_pipe, y_pipe = _resolve_template(t)
            setattr(cls_target, attr_base+'_x', x_pipe)
            setattr(cls_target, attr_base+'_y', y_pipe)

        return lambda y_var=False: (
            getattr(cls_target, attr_base+'_x') if not y_var 
            else getattr(cls_target, attr_base+'_y')
        )

    cls_target.single_pipeline = _process_template(
        cls_target, 
        'single_pipeline_template'
    )
    cls_target.multi_pipeline = _process_template(
        cls_target, 
        'multi_pipeline_template'
    )


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
            class_._check_intent()
            _process_templates(class_)
            _register_intent(class_)
        return class_
