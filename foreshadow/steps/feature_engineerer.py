"""Feature engineering module as a step in Foreshadow workflow."""
from collections import defaultdict

from .preparerstep import PreparerStep
from foreshadow.smart.feature_engineerer import FeatureEngineerer as \
    _FeatureEngineerer


class FeatureEngineererMapper(PreparerStep):
    """Determine and perform best data cleaning step."""

    def __init__(self, **kwargs):
        """Define the single step for FeatureEngineering, using SmartFeatureEngineerer.

        Args:
            *args: args to PreparerStep constructor.
            **kwargs: kwargs to PreparerStep constructor.

        """
        super().__init__(**kwargs)

    def get_mapping(self, X):
        """Maps the columns of the data frame by domain-tag then by Intent
        stored in the ColumnSharer.
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
        """Instead of using i as the outer layer key,
        should we use some more specific like the key
        in columns_by_domain_and_intent, which is ${domain}_${intent}?
        """

        columns_by_domain_and_intent = list(
            columns_by_domain_and_intent.values()
        )

        return self.separate_cols(
            transformers=[
                [_FeatureEngineerer(column_sharer=self.column_sharer)]
                for col_group in columns_by_domain_and_intent
            ],
            cols=columns_by_domain_and_intent,
        )
