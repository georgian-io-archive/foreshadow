# import re
#
# from foreshadow.cleaners.data_cleaner import BaseCleaner
# from foreshadow.core.base import DropMixin
#
#
# def drop_transform(text, return_search=False):
#     """Drop this column at the cleaning stage.
#
#     Args:
#         text: string of text
#
#     Returns:
#         length of match, new string assuming a match.
#         Otherwise: None, original text.
#
#     """
#     regex = "^$"
#     text = str(text)
#     res = re.search(regex, text)
#     if res is not None:
#         res = 1
#     else:
#         res = 0
#     if return_search:
#         return text, res
#     return text
#
#
# class DropCleaner(BaseCleaner, DropMixin):
#     """Clean financial data.
#
#     Note: requires pandas input dataframes.
#
#     """
#
#     def __init__(self):
#         transformations = [drop_transform]
#         super().__init__(transformations)
