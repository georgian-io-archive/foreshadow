import numpy as np
import pandas as pd

from foreshadow.cachemanager import CacheManager
from foreshadow.preparer import FlattenMapper


def test_json_flattening():
    """Test json input are flattened correctly."""

    data = pd.DataFrame(
        {
            "json": [
                '{"date": "2019-04-11"}',
                '{"financial": "$1.0"}',
                '{"financial": "$1000.00"}',
                '{"random": "asdf"}',
            ]
        },
        columns=["json"],
    )
    cs = CacheManager()
    dc = FlattenMapper(cache_manager=cs)
    dc.fit(data)
    transformed_data = dc.transform(data)
    check = pd.DataFrame(
        [
            ["2019-04-11", np.nan, np.nan],
            [np.nan, "$1.0", np.nan],
            [np.nan, "$1000.00", np.nan],
            [np.nan, np.nan, "asdf"],
        ],
        columns=["json_date", "json_financial", "json_random"],
    )
    assert np.all(
        np.equal(
            transformed_data.values[transformed_data.notna()],
            check.values[check.notna()],
        )
    )


def test_json_flattening_with_non_json_columns():
    data = pd.DataFrame(
        {
            "json": [
                '{"date": "2019-04-11"}',
                '{"financial": "$1.0"}',
                '{"financial": "$1000.00"}',
                '{"random": "asdf"}',
            ],
            "num": [1, 2, 3, 4],
        },
        columns=["json", "num"],
    )
    cs = CacheManager()
    dc = FlattenMapper(cache_manager=cs)
    dc.fit(data)
    transformed_data = dc.transform(data)
    check = pd.DataFrame(
        [
            ["2019-04-11", np.nan, np.nan, 1],
            [np.nan, "$1.0", np.nan, 2],
            [np.nan, "$1000.00", np.nan, 3],
            [np.nan, np.nan, "asdf", 4],
        ],
        columns=["json_date", "json_financial", "json_random", "num"],
    )
    assert np.all(
        np.equal(
            transformed_data.values[transformed_data.notna()],
            check.values[check.notna()],
        )
    )
