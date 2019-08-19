"""Classes to be configured by user for customizing parameter tuning."""

from collections import MutableMapping

import foreshadow.serializers as ser


"""
2. cases:

1. Apply override to initial columns

In this case, we simply need to override the get_mapping result. This is
hard to do because it is computed at .fit() time, not __init__ time. We need to
compute it at .fit() time because we need access to the dataset. Instead,
we will pass overrides to the __init__ and handle the errors if users choose
wrong columns.


2. apply override to a dynamically created transformer

In this case, the output from a previous step in the PreparerStep's pipeline
created new columns. Thesee will not be available at get_mapping() time. If
we pass in these columns to ParallelProcessor, it will try to slice then out
which will break. We do however know the initial column and, knowing
DynamicPipeline's naming scheme, the new column's name. We can enable an
override on a per column level by passing in the eventual columns to be
overridden to that group process.



ParamSpec
"""


class ParamSpec(MutableMapping, ser.ConcreteSerializerMixin):
    """Holds the specification of the parameter search space.

    A search space is a dict or list of dicts. This search space should be
    viewed as one run of optimization on the foreshadow object. The
    algorithm for optimization is determined by the optimizer that is
    chosen. Hence, this specification is agnostic of the optimizer chosen.

    A dict represents the set of parameters to be applied in a single run.

    A list represents a set of choices that the algorithm (again, agnostic
    at this point) can pick from.

    For example, imagine s as our top level object, of structure:

        s (object)
            .transformer (object)
                .attr

    s has an attribute that may be optimized and in turn, that object has
    parameters that may be optimized. Below, we try two different
    transformers and try 2 different parameter specifications for each.
    Note that these parameters are specific to the type of transformer
    (StandardScaler does not have the parameter feature_range and vice versa).

    [
            {
                "s__transformer": "StandardScaler",
                "s__transformer__with_mean": [False, True],
            },
            {
                "s__transformer": "MinMaxScaler",
                "s__transformer__feature_range": [(0, 1), (0, 0.5)]
                ),
            },
        ],

    Here, the dicts are used to tell the optimizer where to values to set
    are. The lists showcase the different values that are possible.
    """

    def __init__(self, fs_pipeline=None, X_df=None, y_df=None):
        """Initialize, and if args are passed, auto create param distribution.

        Only pass the init arguments if automatic param spec determination
        is desired.

        Args:
            fs_pipeline: Foreshadow.pipeline
            X_df: input DataFrame of data points
            y_df: input DataFrame of labels

        Raises:
            ValueError: if either all kwargs are not passed or all aren't
                passed.
            NotImplementedError: All kwargs passed.

        """
        if not (fs_pipeline is None) == (X_df is None) == (y_df is None):
            raise ValueError(
                "Either all kwargs are None or all are set. To "
                "use automatic param determination, pass all "
                "kwargs. Otherwise, manual setting can be "
                "accomplished using set_params."
            )
        self._param_set = False
        self.param_distributions = []
        if not (fs_pipeline is None) and (X_df is None) and (y_df) is None:
            raise NotImplementedError("Automatic param spec not implemented")
            # automatic pipelining.
            # params = fs_pipeline.get_params()
            # for kwarg in kwargs:
            #     key, delim, subkey = kwarg.partition('__')
            #     self.param_distribution[key] = {}
            #     while delim !=  '':
            #         pass
            # self._param_set = True

    def get_params(self, deep=True):
        """Get the params for this object. Used for serialization.

        Args:
            deep: Does nothing. Here for sklearn compatibility.

        Returns:
            Members that need to be set for this object.

        """
        return self.param_distributions

    def set_params(self, **params):
        """Set the params for this object. Used for serialization.

        Also used to init this object when automatic tuning is not used.

        Args:
            **params: Members to set from get_params.

        Returns:
              self.

        """
        self.param_distributions = params["param_distributions"]
        self._param_set = True
        return self

    def __call__(self):
        """Overridden for MutableMapping.

        Returns:
            self.param_distributions

        """
        return self.param_distributions

    def __iter__(self):
        """Iterate over self.param_distributions.

        Returns:
            iter(self.param_distributions)

        """
        return iter(self.param_distributions)

    def __getitem__(self, item):
        """Return value at index item from internal list of params.

        Args:
            item: index in list.

        Returns:
            item at index from self.param_distributions.

        """
        return self.param_distributions[item]

    def __setitem__(self, key, value):
        """Set value at index key from internal list of params.

        Args:
            key: index
            value: value

        """
        self.param_distributions[key] = value

    def __len__(self):
        """Length of self.param_distributions list.

        Returns:
            len(self.param_distributions)

        """
        return len(self.param_distributions)

    def __delitem__(self, key):  # overriding abstract method, not to be used.
        """Not implemented, only overrode because it is an abstract method.

        Args:
            key: not used.

        Raises:
            NotImplementedError: If called

        """
        raise NotImplementedError(
            "Abstract method not implemented. Should " "not be called.fl"
        )
