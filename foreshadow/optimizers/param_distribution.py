"""Classes to be configured by user for customizing parameter tuning."""

from hyperopt import hp
import foreshadow.serializers as ser
from collections import MutableMapping


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
"""


class ParamSpec(MutableMapping, ser.ConcreteSerializerMixin):
    def __init__(self, fs_pipeline=None, X_df=None, y_df=None):
        if not (fs_pipeline is None) == (X_df is None) == (y_df is None):
            raise ValueError("Either all kwargs are None or all are set. To "
                             "use automatic param determination, pass all "
                             "kwargs. Otherwise, manual setting can be "
                             "accomplished using set_params.")
        self._param_set = False
        self.param_distribution = []
        if not (fs_pipeline is None) and (X_df is None) and (y_df) is None:
            params = fs_pipeline.get_params()
            for kwarg in kwargs:
                key, delim, subkey = kwarg.partition('__')
                self.param_distribution[key] = {}
                while delim !=  '':
                    pass
            self._param_set = True

    def get_params(self, deep=True):
        return self.param_distribution

    def set_params(self, **params):
        self.param_distribibution = params['param_distribution']
        self._param_set = True

    def __call__(self):
        return self.param_distribibution

    def __getitem__(self, item):
        return self.


if __name__ == '__main__':
    # ParamSpec().to_json("test")
    from foreshadow import Foreshadow
    ParamSpec(Foreshadow())