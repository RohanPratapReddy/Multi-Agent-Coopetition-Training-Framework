"""MACTF telemetry app."""

from __future__ import annotations

from .apps import TelemetryConfig

default_app_config = "mactf.contrib.telemetry.apps.TelemetryConfig"

__all__ = ["TelemetryConfig", "default_app_config"]
