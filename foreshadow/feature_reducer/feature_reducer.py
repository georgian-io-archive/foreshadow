"""Feature Reducer module in Foreshadow workflow."""
from collections import defaultdict

from foreshadow.core.preparerstep import PreparerStep
from foreshadow.transformers.core import SmartTransformer
from foreshadow.transformers.core.notransform import NoTransform


class FeatureReducer(PreparerStep):
    def __init__(self, *args, **kwargs):
        """Define the single step for FeatureReducer, using SmartReducer.

        Args:
            *args: args to PreparerStep initializer.
            **kwargs: kwargs to PreparerStep initializer.

        """
        super().__init__(*args, use_single_pipeline=True, **kwargs)

    def get_mapping(self, X):
        """Return the mapping of transformations for the FeatureReducer step.
        Current code only supports intent-based reduction at this moment.

        Args:
            X: input DataFrame.

        Returns:
            Mapping in accordance with super.

        """

        """
        A longer discussion. Please correct me if I'm wrong.
        Feature reduction could look at columns in (at least) 2 ways:
        1. By intent
        2. All columns as a whole
        3. One after the other? Probably option1 then option2.
        4. Other ways? This requires more research...

        Based on current implementation,
        it is only possible to choose one mapping from option 1 or 2.
        Option 3 may not be possible.

        The reason is that we must provide a predefined column_mapping,
        fixing the column names.

        Assuming that we are using Option 3 with a column_mapping like this:
        {
            0: {
                # columns with categorical intents
                "inputs": ([col1, col2, col3,..., col9], ),
                "steps": [SmartFeatureReducer,],
            },
            1: {
                # columns with numeric intents
                "inputs": ([col10, col11,..., col16], ),
                "steps": [SmartFeatureReducer,],
            },
            2: {
                # all columns
                "inputs": ([col1, col2, col3,..., col16], ),
                "steps": [SmartFeatureReducer,],
            },
        }

        If we choose a reduction method that does not modify column names, t
        his may be fine:

        Say we apply this reduction method is applied to mapping[0]
        and/or mapping [1] and some columns are removed.

        When we process mapping[2], we face the fact of missing column names
        in the dataframe. In this case, we may just do a pre-processing step
        to remove missing columns from mapping[2]["inputs"] and proceed
        as usual.

        However, what if the SmartFeatureReducer decides to use a method
        that not only reduce dimensionality but also modify the name,
        like PCA? In that case, the columns in mapping[3]["inputs"]
        may not be valid. We have to somehow get the latest columns
        from the dataframe first before applying reduction on the whole df.

        To achieve this, it seems that we need to modify the method
        parallelize_smart_steps and/or the class ParallelProcessor
        to inject this column list freshing operation.
        """

        def group_by(iterable, column_sharer_key):
            result = defaultdict(list)
            for col in iterable:
                result[self.column_sharer[column_sharer_key][col]].append(col)
            return result

        columns = X.columns.values.tolist()
        columns_by_intent = group_by(columns, "intent")

        """Not sure where the drop_feature functionality would apply.
        Would reducer produce empty columns? If yes, the concrete reducer
        should check and apply drop column functionality before return.
        """
        column_mapping = {
            i: {
                "inputs": (cols,),
                "steps": [
                    SmartFeatureReducer(column_sharer=self.column_sharer)
                ],
            }
            for i, cols in enumerate(list(columns_by_intent.values()))
        }

        return column_mapping


class SmartFeatureReducer(SmartTransformer):
    """Intelligently decide which feature reduction function
    should be applied."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def pick_transformer(self, X, y=None, **fit_params):
        """Get best transformer for a given set of columns.
        Args:
            X: input DataFrame
            y: input labels
            **fit_params: fit_params
        Returns:
            Best feature engineering transformer.
        """
        return NoTransform(column_sharer=self.column_sharer)
