"""Metrics used across Foreshadow for smart decision making."""

from functools import wraps

class Metric():  # Metric wrapper
    """Metric class for metric functions.

    Apply this class by using the metric decorator.
    """

    def __init__(self, fn, default_return=None, *kwargs):
        """Initialize metric function with fn as attribute.

        Args:
            fn: metric function
        """
        self.fn = fn
        self.default_return = default_return

    def __call__(self, feature, invert=False, **kwargs):
        """Use the metric function passed at initialization.

        Args:
            feature: feature/column of pandas dataset
                requires it.
            **kwargs: any keyword arguments to metric function

        Returns:
            the metric computation defined by the metric.

        """
        # if encoder is not None:  # explicit since encoder is common kwarg.
        #     kwargs["encoder"] = encoder  # passing the encoder through with
        #     # kwargs, where it is guaranteed to not already be a kwarg (as
        #     # it would have been passed through the explicit named argument,
        #     # encoder. The internal self.fn is expected to accept encoder if
        #     # it is passed and it is the users job to only pass encoder when
        #     # it is accepted as a kwarg by self.fn
        try:
            self._last_call = self.fn(feature, **kwargs)
        except Exception as e:
            print(str(e))
            if self.default_return is not None:
                return self.default_return
            else:
                raise e
            
        return self._last_call if not invert else (1. - self._last_call)

    def last_call(self):
        """Value from previous call to metric function.

        Returns:
            last call to metric_fn (self.fn)

        """
        return self._last_call

    def __str__(self):
        """Pretty print.

        Returns:
            $class.$fn

        """
        return "{0}.{1}".format(self.__class__.__name__, self.fn.__name__)

    def __repr__(self):
        """Unambiguous print.

        Returns:
            <$class, $fn, $id>

        """
        return "{0} with function '{1}' object at {2}>".format(
            str(self.__class__)[:-1], self.fn.__name__, id(self)
        )


class metric():
    """Decorate any metric function.

    Args:
        fn: function to decorate.

    Returns:
        Metric function as callable object.

    """
    def __init__(self, default_return=None):
        self.default_return = default_return

    def __call__(self, fn):
        return Metric(fn, self.default_return)

# def metric(default_return=None):
#     def real_decorator(fn):
#         @wraps(fn)
#         def wrapper(*args, **kwargs):
#             wrapped_metric = Metric(fn, default_return=default_return)
#             return wrapped_metric(*args, **kwargs)
#         return wrapper
#     return real_decorator
