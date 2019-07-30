"""Metrics used across Foreshadow for smart decision making."""

from foreshadow.core import logging


class MetricWrapper:
    """MetricWrapper class for metric functions.

    Note:
        Apply this class by using the metric decorator.

    Params:
        fn: Metric function to be wrapped
        default_return (bool): The default return value of the wrapped
            function.

    .. automethod:: __call__

    """

    def __init__(self, fn, default_return=None):
        self.fn = fn
        self.default_return = default_return

    def __call__(self, feature, invert=False, **kwargs):
        """Use the metric function passed at initialization.

        Note:
            If default_return was set, the wrapper will suppress any errors
            raised by the wrapped function.

        Args:
            feature: feature/column of pandas dataset
                requires it.
            invert (bool): Invert the output (1-x)
            **kwargs: any keyword arguments to metric function

        Returns:
            The metric computation defined by the metric.

        Raises:
            re_raise: If default return is not set the metric will display \
                the raised errors in the function.

        """
        try:
            self._last_call = self.fn(feature, **kwargs)
        except Exception as re_raise:
            logging.debug(
                "There was an exception when calling {}".format(self.fn)
            )
            if self.default_return is not None:
                return self.default_return
            else:
                raise re_raise

        return self._last_call if not invert else (1.0 - self._last_call)

    def last_call(self):
        """Value from previous call to metric function.

        Returns:
            Last call to metric_fn (self.fn)

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


class metric:
    """Decorate any metric function.

    Args:
        fn: function to decorate. (Automatically passed in)
        default_return (bool): The default return value of the Metric function.

    Returns:
        Metric function as callable object.

    """

    def __init__(self, default_return=None):
        self.default_return = default_return

    def __call__(self, fn):
        """Get the wrapped metric function.

        Args:
            fn: The metric function to be wrapped.

        Returns:
            An instance `MetricWrapper` that wraps a function.

        """
        return MetricWrapper(fn, self.default_return)
