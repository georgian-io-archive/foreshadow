"""Custom foreshadow metrics for computing data statistics."""

import glob
import inspect
import os


def _get_classes():
    """Return list of classes found in transforms directory.

    Returns:
        list of classes found in transforms directory

    """
    files = glob.glob(os.path.dirname(__file__) + "/*.py")
    imports = [
        os.path.basename(f)[:-3]
        for f in files
        if os.path.isfile(f) and not f.endswith("__init__.py")
    ]
    modules = [
        __import__(i, globals(), locals(), ["object"], 1) for i in imports
    ]
    classes = [
        c[1]
        for m in modules
        for c in inspect.getmembers(m)
        if inspect.isclass(c[1]) or inspect.iscode(c[1])
    ]

    classes = [
        c[1]
        for m in modules
        for c in inspect.getmembers(m)
        if inspect.isclass(c[1]) or callable(c[1])
    ]

    return classes


def _get_modules(metrics, globals_, module_name):
    """Import internal metrics from internals.py or internal directory.

    Assumes that the input classes are all the metrics to be imported.

    Args:
        metrics: A list of metrics (classes or functions).
        globals_: The globals in the callee's context
        module_name: The module name

    Returns:
        The list of wrapped transformers.

     Raises:
           NotImplementedError: if the imported metric is not a supported type.

    """
    metric_names = []

    for m in metrics:
        if inspect.isclass(m):
            copied_m = type(m.__name__, (m, *m.__bases__), dict(m.__dict__))
            copied_m.__module__ = module_name
            mname = copied_m.__name__
            globals_[mname] = copied_m
        elif inspect.isfunction(m):
            mname = m.__name__
            globals_[mname] = m
        elif callable(m):
            mname = m.fn.__name__
            globals_[mname] = m
        else:
            raise NotImplementedError(
                "metric: '{}' of type: '{}' is not "
                "supported.".format(m, type(m))
            )
        metric_names.append(mname)
    return metric_names


classes = _get_modules(_get_classes(), globals(), __name__)
__all__ = classes
