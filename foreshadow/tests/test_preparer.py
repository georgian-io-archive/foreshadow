"""Test the data_preparer.py file."""
import pytest

from foreshadow.utils.testing import get_file_path


@pytest.mark.parametrize(
    "cleaner_kwargs,expected_error",
    [
        ({}, None),
        (None, None),
        ({"random_kwarg": "random_value"}, TypeError),  # replace with real
        # kwargs
        ([], ValueError),
    ],
)
def test_data_preparer_init(cleaner_kwargs, expected_error):
    """Test creation of DataPreparer with kwargs.

    Args:
          cleaner_kwargs: kwargs to CleanerMapper step
          expected_error: expected error from initialization. None if no
            expected error.

    """
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer

    cs = ColumnSharer()
    if expected_error is not None:
        with pytest.raises(expected_error) as e:
            DataPreparer(cs, cleaner_kwargs=cleaner_kwargs)
        assert issubclass(e.type, expected_error)
    else:
        DataPreparer(cs, cleaner_kwargs=cleaner_kwargs)


@pytest.mark.parametrize("cleaner_kwargs", [({}), (None)])
def test_data_preparer_fit(cleaner_kwargs):
    """Test fitting of DataPreparer after creation with kwargs.

    Args:
          cleaner_kwargs: kwargs to CleanerMapper step

    """
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer
    import pandas as pd

    boston_path = get_file_path("data", "boston_housing.csv")
    data = pd.read_csv(boston_path)

    cs = ColumnSharer()
    dp = DataPreparer(cs, cleaner_kwargs=cleaner_kwargs)
    dp.fit(data)


@pytest.mark.parametrize("deep", [True, False])
def test_data_preparer_get_params(deep):
    """Test thet get_params returns the minimum required.

    Args:
        deep: arg to get_params

    """
    from foreshadow.preparer import DataPreparer

    dp = DataPreparer()
    params = dp.get_params(deep=deep)
    assert "cleaner_kwargs" in params
    assert "column_sharer" in params
    assert "engineerer_kwargs" in params
    assert "intent_kwargs" in params
    assert "preprocessor_kwargs" in params
    assert "reducer_kwargs" in params
    assert "y_var" in params
    assert "steps" in params


def test_data_preparer_serialization_has_one_column_sharer():
    """Test DataPreparer serialization after fitting. The serialized
    object should contain only 1 column_sharer instance.

    """
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer
    import pandas as pd

    boston_path = get_file_path("data", "boston_housing.csv")
    data = pd.read_csv(boston_path)

    cs = ColumnSharer()
    dp = DataPreparer(cs)
    dp.fit(data)

    dp_serialized = dp.serialize(method="dict", deep=True)

    key_name = "column_sharer"
    assert key_name in dp_serialized
    dp_serialized.pop(key_name)

    def check_has_no_column_sharer(dat, target):
        if isinstance(dat, dict):
            matching_keys = [key for key in dat if key.endswith(target)]
            assert len(matching_keys) == 0
            for key in dat:
                check_has_no_column_sharer(dat[key], target)
        elif isinstance(dat, list):
            for item in dat:
                check_has_no_column_sharer(item, target)

    check_has_no_column_sharer(dp_serialized, key_name)


def test_data_preparer_deserialization():
    from foreshadow.preparer import DataPreparer
    from foreshadow.columnsharer import ColumnSharer
    import pandas as pd

    boston_path = get_file_path("data", "boston_housing.csv")
    data = pd.read_csv(boston_path)
    # data = data[["crim", "indus", "ptratio", "tax", "zn"]]
    # data = data[["crim", "indus"]]
    # data = data[["nox"]]

    cs = ColumnSharer()
    dp = DataPreparer(cs)

    dp.fit(data)
    data_transformed = dp.transform(data)
    dp.to_json("data_preparerer.json")

    dp2 = DataPreparer.from_json("data_preparerer.json")
    dp2.fit(data)
    data_transformed2 = dp2.transform(data)

    from pandas.util.testing import assert_frame_equal

    assert_frame_equal(data_transformed, data_transformed2)
