"""Sub Numeric Intents."""
# flake8: noqa
#
# import re
#
# from foreshadow.intents.base import PipelineTemplateEntry
# from foreshadow.intents.general import NumericIntent
# from foreshadow.transformers.concrete import DropFeature
# from foreshadow.transformers.smart import (
#     FinancialCleaner,
#     Scaler,
#     SimpleImputer,
# )
#
#
# class FinancialIntent(NumericIntent):
#     """Match financial data.
#     Handles American and European Style numbers. Handles brackets for
#     accounting data.
#     """
#
#     children = []
#
#     single_pipeline_template = [
#         PipelineTemplateEntry("dropper", DropFeature, False),
#         PipelineTemplateEntry("fin_cleaner", FinancialCleaner, True),
#         PipelineTemplateEntry("simple_imputer", SimpleImputer, False),
#         PipelineTemplateEntry("scaler", Scaler, True),
#     ]
#
#     multi_pipeline_template = []
#
#     @classmethod
#     def is_intent(cls, df):
#         """Return true if column contains financial data.
#         .. # noqa: I101
#         .. # noqa: I201
#         """
#         us_num = re.compile(
#             (
#                 r"(?<!\S)(\[|\()?((-(?=[0-9\.]))?([0-9](\,(?=[0-9]{3}))?)*"
#   @adithyabsk
# adithyabsk 3 hours ago  Member
# Don't remember if is here or in Smart but we should snag these regexes before we delete these files.
#
#   @cchoquette
# cchoquette 2 hours ago  Author Member
# rename to old.
#
#  @cchoquette	Reply…
#                 r"((\.(?=[0-9]))|((?<=[0-9]))\.)?[0-9]*)(\)|\])?%?(?!\S)"
#             )
#         )
#         eu_num = re.compile(
#             (
#                 r"(?<!\S)(\[|\()?((-(?=[0-9\,]))?([0-9](\.(?=[0-9]{3}))?)*"
#                 r"((\,(?=[0-9]))|((?<=[0-9]))\,)?[0-9]*)(\)|\])?%?(?!\S)"
#             )
#         )
#
#         data = df.iloc[:, 0].dropna()
#
#         return ((data.str.match(us_num).sum()) / len(data) > 0.2) or (
#             (data.str.match(eu_num).sum()) / len(data) > 0.2
#         )
