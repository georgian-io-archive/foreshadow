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

    def __call__(self, feature, encoder=None, **kwargs):
        """Use the metric function passed at initialization.

        Args:
            feature: feature/column of pandas dataset
            encoder: the encoder being used. Only required if metric fn
                requires it.
            **kwargs: any keyword arguments to metric function

        Returns:
            the metric computation defined by the metric.

        """
        if encoder is not None:  # explicit since encoder is common kwarg.
            kwargs["encoder"] = encoder
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


# ------------------------------------------------
@metric
def unique_count(feature):
    """Count number of unique values in feature.

    Args:
        feature: feature/column of pandas dataset

    Returns:
        number of unique values in feature as int.

    """
    return len(feature.value_counts())


@metric
def unique_count_bias(feature):
    """Difference of count of unique values relative to the length of feature.

    Args:
        feature: feature/column of pandas dataset

    Returns:
        Number of unique values relative to the length of dataset.

    """
    return len(feature) - len(feature.value_counts())


@metric
def unique_count_weight(feature):
    """Normalize count number of unique values relative to length of feature.

    Args:
        feature: feature/column of pandas dataset

    Returns:
        Normalized Number of unique values relative to length of feature.

    """
    return len(feature.value_counts()) / len(feature)


# ------------------------- Example below
# class SmartTransformer(self)
#   def __init__(self, metrics=default, weights=default)
#     self.metrics = metrics
#     self.metrics = {'ohe': [unique_count_weight, delim_diff], 'ur': [],
#                     'dummy': [unique_count, delim_diff]}
#     self.weights = {'ohe': [0.7, 0.3], 'ur': [], 'dummy': [0.3, 0.7]}
#
#   def compute_confidences(self, column, encoders):
#     ovr_confidences = []
#     for enc in encoders:
#       conf = 0
#       for i in range(len(self.metrics)):
#         conf += self.metrics[enc][i](column, encoder=encoder) * \
#                 self.weights[enc][i]
#       ovr_confidences.append(conf)
#     return ovr_confidences
#
#   def _get_transformer(self, X, y=None, **fit_params):
#     data = X.iloc[:, 0]
#
#     confidences = self.compute_confidences(column, self.encoders)
#     best_confidence = max(confidences)
#     if best_confidence > conf_thresh:
#
#     if self.y_var:
#       return LabelEncoder()
#     elif delim_diff(data)[0] < 0:
#       # return DummyEncoder(delim_diff.last_call()[1])
#       return 1
#     elif unique_count(data) <= self.unique_num_cutoff:
#       return 1
#     elif reduce_count(data, self, 2) == 1:
#       return 1
#     else:
#       return 1
