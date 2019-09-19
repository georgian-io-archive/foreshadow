"""Test data_cleaner.py"""


def test_data_cleaner_fit():
    """Test basic fit call."""
    import pandas as pd
    import numpy as np
    from foreshadow.steps import CleanerMapper
    from foreshadow.columnsharer import ColumnSharer

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
    cs = ColumnSharer()
    dc = CleanerMapper(column_sharer=cs)
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
    from foreshadow.columnsharer import ColumnSharer
    import numpy as np

    data = pd.DataFrame(
        {"financials": ["$1.00", "$550.01", "$1234", "$12353.3345"]},
        columns=["financials"],
    )
    cs = ColumnSharer()
    dc = CleanerMapper(column_sharer=cs)
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
    from foreshadow.columnsharer import ColumnSharer
    import numpy as np

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
    cs = ColumnSharer()
    dc = CleanerMapper(column_sharer=cs)
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


def test_drop():
    """Test drop called when expected to."""
    import pandas as pd
    from foreshadow.preparer import CleanerMapper
    from foreshadow.columnsharer import ColumnSharer

    columns = ["financials"]
    data = pd.DataFrame({"financials": ["", "", "", ""]}, columns=columns)
    cs = ColumnSharer()
    dc = CleanerMapper(column_sharer=cs)
    dc.fit(data)
    transformed_data = dc.transform(data)
    assert transformed_data.empty
    assert transformed_data.columns == columns


def test_numerical_input():
    """Test numerical input."""
    import numpy as np
    import pandas as pd
    from foreshadow.preparer import CleanerMapper
    from foreshadow.columnsharer import ColumnSharer

    columns = ["financials"]
    data = pd.DataFrame({"financials": np.arange(10)}, columns=columns)
    cs = ColumnSharer()
    dc = CleanerMapper(column_sharer=cs)
    dc.fit(data)
    transformed_data = dc.transform(data)
    assert np.array_equal(transformed_data, data)


def test_numerical_input_fittransform():
    """Test numerical input."""
    import numpy as np
    import pandas as pd
    from foreshadow.preparer import CleanerMapper
    from foreshadow.columnsharer import ColumnSharer

    columns = ["financials"]
    data = pd.DataFrame({"financials": np.arange(10)}, columns=columns)
    cs = ColumnSharer()
    dc = CleanerMapper(column_sharer=cs)
    transformed_data = dc.fit_transform(data)
    assert np.array_equal(transformed_data, data)


# TODO test graph, could be implemented very wrong.
