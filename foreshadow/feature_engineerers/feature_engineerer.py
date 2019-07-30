"""Feature engineering module as a step in Foreshadow workflow."""
from collections import defaultdict

from foreshadow.core.preparerstep import PreparerStep
from foreshadow.transformers.core import SmartTransformer
from foreshadow.transformers.core.notransform import NoTransform


class FeatureEngineerer(PreparerStep):
    """Determine and perform best data cleaning step."""

    def __init__(self, *args, **kwargs):
        """Define the single step for FeatureEngineering, using SmartFeatureEngineerer.

        Args:
            *args: args to PreparerStep constructor.
            **kwargs: kwargs to PreparerStep constructor.

        """
        super().__init__(*args, use_single_pipeline=False, **kwargs)

    def get_mapping(self, X):
        """TODO as mentioned in the design documentation, feature engineerer
        aggregates columns by domain-tag in the columnsharer and then by
        Intent to decide if any feature generation or reduction is possible.
        """

        def group_by(iterable, column_sharer_key):
            result = defaultdict(list)
            for col in iterable:
                result[self.column_sharer[column_sharer_key][col]].append(col)
            return result

        columns = X.columns.values.tolist()
        columns_by_domain = group_by(columns, "domain")

        columns_by_domain_and_intent = defaultdict(list)
        for domain in columns_by_domain:
            columns_by_intent = group_by(columns_by_domain[domain], "intent")
            for intent in columns_by_intent:
                columns_by_domain_and_intent[
                    domain + "_" + intent
                ] += columns_by_intent[intent]

        """TODO calling separate_cols doesn't seem to work when we want the inputs
        to be a tuple like (['col1', 'col2', ...], ).
        "input": (cols,), seems to fail due to out of index error.
        """

        """Instead of using i as the outer layer key,
        should we use some more specific like the key
        in columns_by_domain_and_intent, which is ${domain}_${intent}?
        """

        return {
            i: {
                "inputs": (cols,),
                "steps": [
                    SmartFeatureEngineerer(column_sharer=self.column_sharer)
                ],
            }
            for i, cols in enumerate(
                list(columns_by_domain_and_intent.values())
            )
        }


class SmartFeatureEngineerer(SmartTransformer):
    """Intelligently decide which feature engineering function
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
