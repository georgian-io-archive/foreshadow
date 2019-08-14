"""Tests for the gui DTO objects."""
from unittest import mock

import pytest


@pytest.mark.parametrize("event_data", [{}, {"invalid_key": "invalid_value"}])
@mock.patch("builtins.open", new_callable=mock.mock_open())
def test_gui_event(mock_open, event_data):
    """Test that the gui event automatically adds timestamp and throws errors.

    Args:
        event_data: the data to load

    """
    from foreshadow.logging.gui import GuiEvent
    from datetime import datetime

    schema = GuiEvent()
    now = datetime.now().strftime("%Y%m%dT%H:%M:%S:%f")
    data = schema.load(event_data).data
    now_time = now[now.find("T") + 1 :]  # noqa: E203 black/flake8 issue
    data_time = data["timestamp"][
        data["timestamp"].find("T") + 1 :  # noqa: E203 black/flake8 issue
    ]
    data_time = int(data_time.replace(":", ""))
    now_time = int(now_time.replace(":", ""))
    assert pytest.approx(data_time, abs=50000) == now_time


@pytest.mark.parametrize(
    "metric_data,expected",
    [
        (
            {
                "smart_decider": "smart_name",
                "selected_transformer": "transformer_name",
                "stats": [{"stat1": "val1"}, {"stat1": "val1"}],
            },
            {},
        ),
        (
            {"stat_name": "test", "stat_value": 1},
            {"stat_name": "test", "stat_value": 1},
        ),
    ],
)
@mock.patch("builtins.open", new_callable=mock.mock_open())
def test_metric_schema(mock_open, metric_data, expected):
    """Test that a metric can be serialized.

    Args:
        metric_data: the data to load
        expected: expected output from schema.load

    """
    from foreshadow.logging.gui import MetricSchema

    schema = MetricSchema()
    data = schema.load(metric_data).data
    assert data == expected


@pytest.mark.parametrize(
    "event_data",
    [
        {
            "smart_transformer": "smart_name",
            "decision": "transformer_name",
            "stats": [
                {"stat_name": "name1", "stat_value": 1},
                {"stat_name": "name2", "stat_value": 2},
            ],
        }
    ],
)
@mock.patch("builtins.open", new_callable=mock.mock_open())
def test_smart_decision(mock_open, event_data):
    """Test that the smart decision

    Args:
        event_data: the data to load

    """
    from foreshadow.logging.gui import SmartDecisionSchema

    schema = SmartDecisionSchema()
    data = schema.load(event_data).data
    assert {
        key: val for key, val in data.items() if key in event_data
    } == event_data
