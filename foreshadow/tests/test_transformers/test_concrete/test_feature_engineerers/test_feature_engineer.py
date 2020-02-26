# """Test feature_engineerer.py"""
# TODO temporarily turning it off since we are not using it and it's
#  dragging down code coverage.
# import numpy as np
# import pandas as pd
# import pytest
#
# from foreshadow.cachemanager import CacheManager
# from foreshadow.smart.feature_engineerer import FeatureEngineerer
# from foreshadow.steps import FeatureEngineererMapper
# from foreshadow.steps.preparerstep import PreparerMapping
#
#
# @pytest.mark.skip("Feature Engineerer is not enabled yet.")
# def test_feature_engineerer_fit():
#     """Test basic fit call."""
#     data = pd.DataFrame(
#         {
#             "age": [10, 20, 33, 44],
#             "weights": [20, 30, 50, 60],
#             "financials": ["$1.00", "$550.01", "$1234", "$12353.3345"],
#         },
#         columns=["age", "weights", "financials"],
#     )
#     cs = CacheManager()
#     cs["domain", "age"] = "personal"
#     cs["domain", "weights"] = "personal"
#     cs["domain", "financials"] = "financial"
#
#     cs["intent", "age"] = "Numeric"
#     cs["intent", "weights"] = "Numeric"
#     cs["intent", "financials"] = "Numeric"
#
#     fem = FeatureEngineererMapper(cache_manager=cs)
#     fem.fit(data)
#     transformed_data = fem.transform(data)
#     assert np.all(
#         np.equal(
#             data.values[data.notna()],
#             transformed_data.values[transformed_data.notna()],
#         )
#     )
#
#
# @pytest.mark.skip("Feature Engineerer is not enabled yet.")
# def test_feature_engineerer_get_mapping():
#     """Test basic fit call."""
#     data = pd.DataFrame(
#         {
#             "age": [10, 20, 33, 44],
#             "weights": [20, 30, 50, 60],
#             "financials": ["$1.00", "$550.01", "$1234", "$12353.3345"],
#         },
#         columns=["age", "weights", "financials"],
#     )
#
#     cs = CacheManager()
#     cs["domain", "age"] = "personal"
#     cs["domain", "weights"] = "personal"
#     cs["domain", "financials"] = "financial"
#
#     cs["intent", "age"] = "Numeric"
#     cs["intent", "weights"] = "Numeric"
#     cs["intent", "financials"] = "Numeric"
#
#     fem = FeatureEngineererMapper(cache_manager=cs)
#     column_mapping = fem.get_mapping(data)
#
#     check_pm = PreparerMapping()
#     check_pm.add(
#         ["age", "weights"],
#         [FeatureEngineerer(cache_manager=cs)],
#         "personal_Numeric",
#     )
#     check_pm.add(
#         ["financials"],
#         [FeatureEngineerer(cache_manager=cs)],
#         "financial_Numeric",
#     )
#
#     for key in column_mapping.store:
#         assert key in check_pm.store
#         assert str(column_mapping.store[key]) == str(check_pm.store[key])
