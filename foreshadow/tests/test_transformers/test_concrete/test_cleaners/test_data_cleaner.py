"""Test data_cleaner.py"""
import pytest


def test_data_cleaner_transform_before_fit():
    import pandas as pd
    from foreshadow.steps import CleanerMapper
    from foreshadow.cachemanager import CacheManager

    data = pd.DataFrame(
        {"financials": ["$1.00", "$550.01", "$1234", "$12353.3345"]},
        columns=["financials"],
    )
    cs = CacheManager()
    dc = CleanerMapper(cache_manager=cs)

    with pytest.raises(ValueError) as e:
        dc.transform(data)

    assert str(e.value) == "Cleaner has not been fitted yet."


# TODO: This is no longer valid as we have separated data cleaner into
#  flattener and cleaner
@pytest.mark.skip("TODO: need to fix the flattener and cleaner issue.")
def test_data_cleaner_fit():
    """Test basic fit call."""
    import pandas as pd
    import numpy as np
    from foreshadow.steps import CleanerMapper
    from foreshadow.cachemanager import CacheManager

    data = pd.DataFrame(
        {
            "dates": ["2019-02-11", "2019/03/12", "2000-04-15", "1900/01/55"],
            "json": [
                '{"date": "2019-04-11"}',
                '{"financial": "$1.0"}',
                '{"financial": "$1000.00"}',
                '{"random": "asdf"}',
            ],
            "financials": ["$1.00", "$550.01", "$1234", "$12353.3345"],
        },
        columns=["dates", "json", "financials"],
    )
    cs = CacheManager()
    dc = CleanerMapper(cache_manager=cs)
    dc.fit(data)
    data = dc.transform(data)
    check = pd.DataFrame(
        [
            ["2019", "02", "11", "2019", "04", "11", np.nan, np.nan, "1.00"],
            ["2019", "03", "12", np.nan, "", "", "1.0", np.nan, "550.01"],
            ["2000", "04", "15", np.nan, "", "", "1000.00", np.nan, "1234"],
            ["1900", "01", "55", np.nan, "", "", np.nan, "asdf", "12353.3345"],
        ],
        columns=[
            "dates0",
            "dates1",
            "dates2",
            "json_date0",
            "json_date1",
            "json_date2",
            "json_financial",
            "json_random",
            "financials",
        ],
    )
    print(data.values)
    print(check.values)
    assert np.all(
        np.equal(data.values[data.notna()], check.values[check.notna()])
    )


def test_financials():
    """Test financial column cleaned correctly."""
    import pandas as pd
    from foreshadow.preparer import CleanerMapper
    from foreshadow.cachemanager import CacheManager
    import numpy as np

    data = pd.DataFrame(
        {"financials": ["$1.00", "$550.01", "$1234", "$12353.3345"]},
        columns=["financials"],
    )
    cs = CacheManager()
    dc = CleanerMapper(cache_manager=cs)
    dc.fit(data)
    transformed_data = dc.transform(data)
    check = pd.DataFrame(
        {"financials": ["1.00", "550.01", "1234", "12353.3345"]},
        columns=["financials"],
    )
    assert np.all(
        np.equal(
            transformed_data.values[data.notna()], check.values[check.notna()]
        )
    )


def test_json():
    """Test json input cleaned correctly."""
    import pandas as pd
    from foreshadow.preparer import CleanerMapper
    from foreshadow.cachemanager import CacheManager
    import numpy as np

    data = pd.DataFrame(
        [
            ["2019-04-11", np.nan, np.nan],
            [np.nan, "$1.0", np.nan],
            [np.nan, "$1000.00", np.nan],
            [np.nan, np.nan, "asdf"],
        ],
        columns=["json_date", "json_financial", "json_random"],
    )
    cs = CacheManager()
    dc = CleanerMapper(cache_manager=cs)
    dc.fit(data)
    data = dc.transform(data)
    check = pd.DataFrame(
        [
            ["2019", "04", "11", np.nan, np.nan],
            [np.nan, "", "", "1.0", np.nan],
            [np.nan, "", "", "1000.00", np.nan],
            [np.nan, "", "", np.nan, "asdf"],
        ],
        columns=[
            "json_date0",
            "json_date1",
            "json_date2",
            "json_financial",
            "json_random",
        ],
    )
    assert np.all(
        np.equal(data.values[data.notna()], check.values[check.notna()])
    )


def test_drop_entire_data_frame():
    """Test drop called when expected to."""
    import pandas as pd
    from foreshadow.preparer import CleanerMapper
    from foreshadow.cachemanager import CacheManager

    columns = ["financials"]
    data = pd.DataFrame({"financials": ["", "", "", ""]}, columns=columns)
    cs = CacheManager()
    dc = CleanerMapper(cache_manager=cs)
    import pytest

    with pytest.raises(ValueError) as excinfo:
        dc.fit_transform(data)
    error_msg = (
        "All columns are dropped since they all have over 90% of "
        "missing values. Aborting foreshadow."
    )
    assert error_msg in str(excinfo.value)


def test_drop_empty_columns():
    """Test drop empty columns called when expected to."""
    import pandas as pd
    from foreshadow.preparer import CleanerMapper
    from foreshadow.cachemanager import CacheManager

    columns = ["financials", "nums"]
    data = pd.DataFrame(
        {"financials": ["", "", "", ""], "nums": [1, 2, 3, 4]}, columns=columns
    )
    cs = CacheManager()
    dc = CleanerMapper(cache_manager=cs)

    transformed_data = dc.fit_transform(data)
    assert len(transformed_data.columns) == 1
    assert list(transformed_data.columns)[0] == "nums"


def test_numerical_input():
    """Test numerical input."""
    import numpy as np
    import pandas as pd
    from foreshadow.preparer import CleanerMapper
    from foreshadow.cachemanager import CacheManager

    columns = ["financials"]
    data = pd.DataFrame({"financials": np.arange(10)}, columns=columns)
    cs = CacheManager()
    dc = CleanerMapper(cache_manager=cs)
    dc.fit(data)
    transformed_data = dc.transform(data)
    assert np.array_equal(transformed_data, data)


def test_numerical_input_fittransform():
    """Test numerical input."""
    import numpy as np
    import pandas as pd
    from foreshadow.preparer import CleanerMapper
    from foreshadow.cachemanager import CacheManager

    columns = ["financials"]
    data = pd.DataFrame({"financials": np.arange(10)}, columns=columns)
    cs = CacheManager()
    dc = CleanerMapper(cache_manager=cs)
    transformed_data = dc.fit_transform(data)
    assert np.array_equal(transformed_data, data)


# TODO test graph, could be implemented very wrong.
