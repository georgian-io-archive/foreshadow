"""Test the data_preparer.py file."""
import pytest


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
          cleaner_kwargs: kwargs to DataCleaner step
          expected_error: expected error from initialization. None if no
            expected error.

    """
    from foreshadow.core.data_preparer import DataPreparer
    from foreshadow.core.column_sharer import ColumnSharer

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
          cleaner_kwargs: kwargs to DataCleaner step

    """
    from foreshadow.core.data_preparer import DataPreparer
    from foreshadow.core.column_sharer import ColumnSharer

    cs = ColumnSharer()
    dp = DataPreparer(cs, cleaner_kwargs=cleaner_kwargs)
    dp.fit([])
