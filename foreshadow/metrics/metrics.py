"""Metrics used across Foreshadow for smart decision making."""


class Metric(object):  # Metric wrapper
    """Metric class for metric functions.

    Apply this class by using the metric decorator.
    """

    def __init__(self, fn):
        """Initialize metric function with fn as attribute.

        Args:
            fn: metric function
        """
        self.fn = fn

    def __call__(self, feature, **kwargs):
        """Use the metric function passed at initialization.

        Note: encoder is an explicit named argument as it will be important
        for many metrics.

        Args:
            feature: feature/column of pandas dataset
            encoder: the encoder being used. Only required if metric fn
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
        self._last_call = self.fn(feature, **kwargs)
        return self._last_call

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


def metric(fn):
    """Decorate any metric function.

    Args:
        fn: function to decorate.

    Returns:
        Metric function as callable object.

    """
    return Metric(fn)
