"""Gui Schem's to ensure proper serialized data is in expected form."""
from datetime import datetime as _datetime

from marshmallow import Schema as _Schema, fields as _fields


class GuiEvent(_Schema):
    """A lightweight object to define the required fields for anything to GUI.

    Automatically adds a timestamp at the time of creation of this object.
    """

    timestamp = _fields.String(
        default=_datetime.now().strftime("%Y%m%dT%H:%M:%S:%f"),
        missing=_datetime.now().strftime("%Y%m%dT%H:%M:%S:%f"),
    )


class MetricSchema(_Schema):
    """Schema for any metric/statistic to be sent to GUI."""

    stat_name = _fields.String(required=True)
    stat_value = _fields.Float(required=True)


class SmartDecisionSchema(GuiEvent):
    """Schema for any Decisions made by SmartTransformers."""

    decision = _fields.String(required=True)
    stats = _fields.List(_fields.Nested(MetricSchema))
    smart_transformer = _fields.String(required=True)
