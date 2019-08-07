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
