"""General intents."""
# flake8: noqa
# from collections import OrderedDict
#
# import numpy as np
# import pandas as pd
# from pandas.api.types import is_numeric_dtype, is_string_dtype
#
# from foreshadow.intents.base import BaseIntent, PipelineTemplateEntry
# from foreshadow.transformers.concrete import DropFeature
# from foreshadow.transformers.smart import (
#     CategoricalEncoder,
#     MultiImputer,
#     Scaler,
#     SimpleImputer,
#     TextEncoder,
# )
#
#
# def _mode_freq(s, count=10):
#     """Compute the mode and the most frequent values.
#     Args:
#         s (:obj:`Series <pandas.Series>`): the series to analyze
#         count (int): the n number of most frequent values
#     Returns:
#         A tuple with the list of modes and (the 10 most common values,
#         their frequency counts, % frequencies)
#     """
#     mode = s.mode().values.tolist()
#     vc = s.value_counts().nlargest(count).reset_index()
#     vc["PCT"] = vc.iloc[:, -1] / s.size
#     return (mode, vc.values.tolist())
#
#
# def _outliers(s, count=10):
#     """Compute the mode and the most frequent values.
#     Args:
#         s (:obj:`Series <pandas.Series>`): the series to analyze
#         count (int): the n largest (magnitude) outliers
#     Returns:
#         a :obj:`Series <pandas.Series>` of outliers
#     """
#     out_ser = s[np.abs(s - s.mean()) > (3 * s.std())]
#     out_df = out_ser.to_frame()
#     out_df["selector"] = out_ser.abs()
#
#     return out_df.loc[out_df["selector"].nlargest(count).index].iloc[:, 0]
#
#
# def _standard_col_summary(df):
#     data = df.iloc[:, 0]
#     nan_num = int(data.isnull().sum())
#     mode, top10 = _mode_freq(data)
#
#     return OrderedDict([("nan", nan_num), ("mode", mode), ("top10", top10)])
#
#
# class GenericIntent(BaseIntent):
#     """See base class.
#     Serves as root of Intent tree. In the case that no other intent applies
#     this intent will serve as a placeholder.
#     """
#
#     children = ["TextIntent", "NumericIntent", "CategoricalIntent"]
#     """Match to CategoricalIntent over NumericIntent"""
#
#     single_pipeline_template = []
#     """No transformers"""
#
#     multi_pipeline_template = [
#         PipelineTemplateEntry("multi_impute", MultiImputer, False)
#     ]
#     """Perform multi imputation over the entire DataFrame."""
#
#     @classmethod
#     def is_intent(cls, df):
#         """Return true by default such that a column must match this.
#         .. # noqa: I101
#         .. # noqa: I201
#         """
#         return True
#
#     @classmethod
#     def column_summary(cls, df):
#         """No statistics can be computed for a general column.
#         .. # noqa: I101
#         .. # noqa: I201
#         """
#         return {}
#
#
# class NumericIntent(GenericIntent):
#     """See base class.
#     Matches to features with numerical data.
#     """
#
#     children = []
#     """No children"""
#
#     single_pipeline_template = [
#         PipelineTemplateEntry("dropper", DropFeature, False),
#         PipelineTemplateEntry("simple_imputer", SimpleImputer, False),
#         PipelineTemplateEntry("scaler", Scaler, True),
#     ]
#     """Perform imputation and scaling using Smart Transformers"""
#
#     multi_pipeline_template = []
#     """No multi pipeline"""
#
#     @classmethod
#     def is_intent(cls, df):
#         """Return true if data is numeric according to pandas.
#
#         .. # noqa: I101
#         .. # noqa: I201
#         """
#         return (
#             not pd.to_numeric(df.iloc[:, 0], errors="coerce")
#             .isnull()
#             .values.ravel()
#             .all()
#         )
#
#     @classmethod
#     def column_summary(cls, df):
#         """Return computed statistics for a NumericIntent column.
#         The following are computed:
#         | nan: count of nans pass into dataset
#         | invalid: number of invalid values after converting to numeric
#         | mean: -
#         | std: -
#         | min: -
#         | 25th: 25th percentile
#         | median: -
#         | 75th: 75th percentile
#         | max: -
#         | mode: mode or np.nan if data is mostly unique
#         | top10: top 10 most frequent values or empty array if mostly \
#             unique [(value, count),...,]
#         | 10outliers: largest 10 outliers
#         .. # noqa: I101
#         .. # noqa: I201
#         """
#         data = df.iloc[:, 0]
#         nan_num = int(data.isnull().sum())
#         invalid_num = int(
#             pd.to_numeric(df.iloc[:, 0], errors="coerce").isnull().sum()
#             - nan_num
#         )
#         outliers = _outliers(data).values.tolist()
#         mode, top10 = _mode_freq(data)
#
#         return OrderedDict(
#             [
#                 ("nan", nan_num),
#                 ("invalid", invalid_num),
#                 ("mean", float(data.mean())),
#                 ("std", float(data.std())),
#                 ("min", float(data.min())),
#                 ("25th", float(data.quantile(0.25))),
#                 ("median", float(data.quantile(0.5))),
#                 ("75th", float(data.quantile(0.75))),
#                 ("max", float(data.max())),
#                 ("mode", mode),
#                 ("top10", top10),
#                 ("10outliers", outliers),
#             ]
#         )
#
#
# class CategoricalIntent(GenericIntent):
#     """See base class.
#     Matches to features with low enough variance that encoding should be used.
#     """
#
#     children = []
#     """No children"""
#
#     single_pipeline_template = [
#         PipelineTemplateEntry("dropper", DropFeature, False),
#         PipelineTemplateEntry("impute_encode", CategoricalEncoder, True),
#     ]
#     """Encode the column automatically"""
#
#     multi_pipeline_template = []
#     """No multi pipeline"""
#
#     @classmethod
#     def is_intent(cls, df):
#         """Return true if the majority of data is categorical by uniqueness.
#         .. # noqa: I101
#         .. # noqa: I201
#         """
#         data = df.iloc[:, 0]
#         if not is_numeric_dtype(data.dtype):
#             return True
#         else:
#             return (1.0 * data.nunique() / data.count()) < 0.2
#
#     @classmethod
#     def column_summary(cls, df):
#         """Compute statistics for a CategoricalIntent column.
#         The following are statistics are computed:
#         | nan: count of nans pass into dataset
#         | mode: mode or np.nan if data is mostly unique
#         | top10: top 10 most frequent values or empty array if mostly unique
#             ``[(value, count),...,]``
#         .. # noqa: I101
#         .. # noqa: I201
#         """
#         return _standard_col_summary(df)
#
#
# class TextIntent(GenericIntent):
#     """See base class.
#     All features can be treated as text.
#     """
#
#     children = []
#     """No children"""
#
#     single_pipeline_template = [
#         PipelineTemplateEntry("text", TextEncoder, False)
#     ]
#     """Encodes the column automatically"""
#
#     multi_pipeline_template = []
#     """No multi pipeline"""
#
#     @classmethod
#     def is_intent(cls, df):
#         """Every column can be interpreted as a text.
#         .. # noqa: I101
#         .. # noqa: I201
#         """
#         data = df.iloc[:, 0]
#         if is_string_dtype(data.dtype):
#             return True
#         else:
#             return False
#
#     @classmethod
#     def column_summary(cls, df):
#         """Return standard computed statistics for a TextIntent column.
#         The following are computed:
#         | nan: count of nans pass into dataset
#         | mode: mode or np.nan if data is mostly unique
#         | top10: top 10 most frequent values or empty array if mostly \
#             unique [(value, count),...,]
#         .. # noqa: I101
#         .. # noqa: I201
#         """
#         return _standard_col_summary(df)
