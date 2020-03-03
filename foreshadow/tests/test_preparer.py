"""Test the data_preparer.py file."""
import pytest

from foreshadow.concrete import NoTransform
from foreshadow.smart import CategoricalEncoder
from foreshadow.utils import ProblemType
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
    from foreshadow.cachemanager import CacheManager

    cs = CacheManager()
    if expected_error is not None:
        with pytest.raises(expected_error) as e:
            DataPreparer(cs, cleaner_kwargs=cleaner_kwargs)
        assert issubclass(e.type, expected_error)
    else:
        DataPreparer(cs, cleaner_kwargs=cleaner_kwargs)


@pytest.mark.parametrize("problem_type", [None, "Unknown"])
def test_data_preparer_y_variable_invalid_problem_type(problem_type):
    from foreshadow.preparer import DataPreparer

    with pytest.raises(ValueError) as e:
        DataPreparer(y_var=True, problem_type=problem_type)
    assert "Invalid Problem Type" in str(e.value)


@pytest.mark.parametrize(
    "problem_type", [ProblemType.CLASSIFICATION, ProblemType.REGRESSION]
)
def test_data_preparer_y_variable(problem_type):
    from foreshadow.preparer import DataPreparer

    dp = DataPreparer(y_var=True, problem_type=problem_type)
    assert len(dp.steps) == 1
    if problem_type == ProblemType.REGRESSION:
        assert isinstance(dp.steps[0][1], NoTransform)
    else:
        assert isinstance(dp.steps[0][1], CategoricalEncoder)


@pytest.mark.parametrize("cleaner_kwargs", [({}), (None)])
def test_data_preparer_fit(cleaner_kwargs):
    """Test fitting of DataPreparer after creation with kwargs.

    Args:
          cleaner_kwargs: kwargs to CleanerMapper step

    """
    from foreshadow.preparer import DataPreparer
    from foreshadow.cachemanager import CacheManager
    import pandas as pd

    boston_path = get_file_path("data", "boston_housing.csv")
    data = pd.read_csv(boston_path)

    cs = CacheManager()
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
    assert "cache_manager" in params
    assert "engineerer_kwargs" in params
    assert "intent_kwargs" in params
    assert "preprocessor_kwargs" in params
    assert "reducer_kwargs" in params
    assert "y_var" in params
    assert "steps" in params


# def test_data_preparer_intent_resolving():
#     from foreshadow.preparer import DataPreparer
#     from foreshadow.cachemanager import CacheManager
#     import pandas as pd
#
#     # from foreshadow.intents import IntentType
#     # from foreshadow.utils import AcceptedKey, Override
#
#     data_path = get_file_path("data", "adult_small.csv")
#     data = pd.read_csv(data_path)
#
#     cs = CacheManager()
#     # cs[AcceptedKey.OVERRIDE][
#     #     "_".join([Override.INTENT, 'age'])
#     # ] = IntentType.CATEGORICAL
#
#     dp = DataPreparer(cs)
#
#     # data["crim"] = np.nan
#
#     dp.fit(data)
#     res = dp.transform(data)
#     print(cs["intent"])
#     print(res)
