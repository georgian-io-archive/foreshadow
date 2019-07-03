import pytest
import re
import pandas as pd


simple_dataframe = pd.Series([i for i in range(10)])


@pytest.mark.parametrize(
    'metric_fn',
    [
        (lambda x: 1),
        (lambda x, y: 1,),
        (lambda x, y, z: 1)
    ]
)
def test_metric_decorate(metric_fn):
    """Test arbitrary function can be converted to Metric using metric.

    Args:
        metric_fn: arbitrary metric function

    """
    from foreshadow.metrics import metric
    from foreshadow.metrics import Metric
    metric_fn = metric(metric_fn)  # applying decorator
    assert isinstance(metric_fn, Metric)


@pytest.mark.parametrize(
    'metric_fn,arg,kwargs',
    [
        (lambda x: 1, 1, {}),
        (lambda x, y: 1, 1, {'y': 1}),
        (lambda x, encoder: 1, 1, {'encoder': 1})
    ]
)
def test_metric_call(metric_fn, arg, kwargs):
    """Test arbitrary function can be called after being Metric using metric.

    Args:
        metric_fn: arbitrary metric function
        arg: arg to metric call
        kwargs: any kwargs to metric call

    """
    from foreshadow.metrics import metric
    metric_fn = metric(metric_fn)
    assert metric_fn(arg, **kwargs) == 1


@pytest.mark.parametrize(
    'metric_fn,arg,kwargs',
    [
        (lambda x: 1, 1, {}),
        (lambda x, y: 1, 1, {'y': 1}),
        (lambda x, encoder: 1, 1, {'encoder': 1})
    ]
)
def test_metric_last_call(metric_fn, arg, kwargs):
    """Test arbitrary function reroutes from call to last_call

    Args:
        metric_fn: arbitrary metric function
        arg: arg to metric call
        kwargs: any kwargs to metric call

    """
    from foreshadow.metrics import metric
    metric_fn = metric(metric_fn)
    _ = metric_fn(arg, **kwargs)
    assert metric_fn.last_call() == 1

@pytest.mark.parametrize(
    'fn,regex',
    [
        ('__repr__', 'class.*Metric'),
        ('__repr__', 'at.*'),
        ('__repr__', 'function.*lambda'),
        ('__str__', '(Metric)'),
        ('__str__', '(lambda)')
    ]
)
def test_metric_print(fn, regex):
    """Test metric prints correct/useful information about itself.

    Args:
        fn: function of metric that returns a string
        regex: useful information to check

    """
    from foreshadow.metrics import Metric
    metric_fn = Metric(lambda x: 1)
    assert re.search(regex, getattr(metric_fn, fn)())


@pytest.mark.parametrize(
    'column,ret',
    [
        (simple_dataframe, simple_dataframe.shape[0])
    ]
)
def test_unique_count(column, ret):
    """Test the unique_count metric.

    Args:
        column (pandas.DataFrame): the input column
        ret: expected unique_count value

    """
    from foreshadow.metrics import unique_count
    assert unique_count(column) == ret


@pytest.mark.parametrize(
    'column,ret',
    [
        (simple_dataframe, 0)
    ]
)
def test_unique_count_bias(column, ret):
    """ Test the unique_count_bias metric.

    Args:
        column (pandas.DataFrame): the input column
        ret: expected unique_count value

    """
    from foreshadow.metrics import unique_count_bias
    assert unique_count_bias(column) == ret


@pytest.mark.parametrize(
    'column,ret',
    [
        (simple_dataframe, 1)
    ]
)
def test_unique_count_weight(column, ret):
    """ Test the unique_count_weight metric.

        Args:
            column (pandas.DataFrame): the input column
            ret: expected unique_count value

        """
    from foreshadow.metrics import unique_count_weight
    assert unique_count_weight(column) == ret
