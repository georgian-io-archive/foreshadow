"""
Sub Numeric Intents
"""

import re

from .base import PipelineTemplateEntry, TransformerEntry
from .general import NumericIntent

from ..transformers.internals import DropFeature
from ..transformers.smart import FinancialCleaner, SimpleImputer, Scaler


class FinancialIntent(NumericIntent):
    """Matches financial data.

    Handles American and European Style numbers. Handles brackets for accounting
    data.

    """

    children = []

    single_pipeline_template = [
        PipelineTemplateEntry("dropper", DropFeature, False),
        PipelineTemplateEntry("fin_cleaner", FinancialCleaner, True),
        PipelineTemplateEntry("simple_imputer", SimpleImputer, False),
        PipelineTemplateEntry("scaler", Scaler, True),
    ]
    """No transformers"""

    multi_pipeline_template = []
    """Performs multi imputation over the entire DataFrame"""

    @classmethod
    def is_intent(cls, df):
        """Returns true by default such that a column must match this"""
        us_num = re.compile(
            r"(?<!\S)(\[|\()?((-(?=[0-9\.]))?([0-9](\,(?=[0-9]{3}))?)*((\.(?=[0-9]))|((?<=[0-9]))\.)?[0-9]*)(\)|\])?%?(?!\S)"
        )
        eu_num = re.compile(
            r"(?<!\S)(\[|\()?((-(?=[0-9\,]))?([0-9](\.(?=[0-9]{3}))?)*((\,(?=[0-9]))|((?<=[0-9]))\,)?[0-9]*)(\)|\])?%?(?!\S)"
        )

        data = df.iloc[:, 0].dropna()

        return ((data.str.match(us_num).sum()) / len(data) > 0.2) or (
            (data.str.match(eu_num).sum()) / len(data) > 0.2
        )
