"""Classes to be configured by user for customizing parameter tuning."""

import foreshadow as fs
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
"""


class ParamSpec(ser.ConcreteSerializerMixin):
    def __init__(self, fs_pipeline, X_df, Y_df):
        self.fs_pipeline = fs_pipeline
        params = self.fs_pipeline.get_params()
        print(params)

    def get_params(self, deep=True):


    def set_params(self, **params):
        pass



if __name__ == '__main__':
    # ParamSpec().to_json("test")
    from foreshadow import Foreshadow
    ParamSpec(Foreshadow())