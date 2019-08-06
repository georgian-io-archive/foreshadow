"""Test preparersteps.py."""

import pytest

from foreshadow.utils.testing import dynamic_import


@pytest.fixture()
def step():
    """Get a PreparerStep subclass instance.

    Note:
        Always returns StandardScaler.

    """
    from foreshadow.steps.preparerstep import PreparerStep
    from foreshadow.steps.autointentmap import AutoIntentMixin

    class Step(PreparerStep, AutoIntentMixin):
        pass

    yield Step


@pytest.mark.parametrize("column_sharer", [True, False])
def test_init_columnsharer(step, column_sharer):
    """Test columnsharer properly init on step.

    Args:
        step: fixture
        column_sharer: True to pass it in, False to not.

    Returns:

    """
    cs = None
    if column_sharer:
        cs = dynamic_import("ColumnSharer", "foreshadow.columnsharer")()
    step = step(column_sharer=cs)
    assert step.column_sharer is not None
