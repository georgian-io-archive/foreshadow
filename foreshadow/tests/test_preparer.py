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


@pytest.mark.parametrize("cleaner_kwargs", [({}), (None)])
def test_data_preparer_serialization(cleaner_kwargs):
    """Test fitting of DataPreparer after creation with kwargs.

    Args:
          cleaner_kwargs: kwargs to CleanerMapper step

    """
    pass
    # from foreshadow.preparer import DataPreparer
    # from foreshadow.columnsharer import ColumnSharer
    # import pandas as pd
    #
    # boston_path = get_file_path("data", "boston_housing.csv")
    # data = pd.read_csv(boston_path)
    #
    # cs = ColumnSharer()
    # dp = DataPreparer(cs, cleaner_kwargs=cleaner_kwargs)
    # dp.fit(data)
    #
    # cs.to_json("column_sharer.json", deep=True)
    # cs2 = ColumnSharer.from_json("column_sharer.json")
    #
    # assert cs == cs2

    # dp.to_json("data_preparerer_deep_true3.json", deep=True)
    # dp.to_yaml("data_preparerer_deep_true2.yaml", deep=True)

    # dp2 = DataPreparer.from_json("data_preparerer_deep_true2.json")
