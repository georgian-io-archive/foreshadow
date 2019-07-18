"""Gui Schema's to ensure proper serialized data is in expected form."""
from datetime import datetime

from marshmallow import Schema, fields


class GuiEvent(Schema):
    """A lightweight object to define the required fields for anything to GUI.

    Automatically adds a timestamp at the time of creation of this object.
    """

    timestamp = fields.String(
        default=datetime.now().strftime("%Y%m%dT%H:%M:%S:%f"),
        missing=datetime.now().strftime("%Y%m%dT%H:%M:%S:%f"),
    )


class MetricSchema(Schema):
    """Schema for any metric/statistic to be sent to GUI."""

    stat_name = fields.String(required=True)
    stat_value = fields.Float(required=True)


class SmartDecisionSchema(GuiEvent):
    """Schema for any Decisions made by SmartTransformers."""

    decision = fields.String(required=True)
    stats = fields.List(fields.Nested(MetricSchema))
    smart_transformer = fields.String(required=True)
