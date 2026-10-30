"""The telemetry contrib app: logging plus a signal-driven episode log."""

from __future__ import annotations

import logging

from ...apps import AppConfig
from ...telemetry import configure_logging


class TelemetryConfig(AppConfig):
    name = "mactf.contrib.telemetry"
    label = "mactf_telemetry"
    verbose_name = "MACTF Telemetry"

    def ready(self) -> None:
        from ... import signals
        from ...conf import get_settings

        settings = get_settings()
        telemetry = settings["TELEMETRY"]
        try:
            log_file = settings.path("LOGS") / "mactf.log"
        except Exception:  # noqa: BLE001 - logging must never block startup
            log_file = None
        configure_logging(
            level=str(telemetry.get("LEVEL", "INFO")),
            json_output=bool(telemetry.get("JSON", False)),
            console=bool(telemetry.get("CONSOLE", True)),
            log_file=log_file,
            force=True,
        )
        logger = logging.getLogger("mactf.telemetry")

        def on_episode_scored(sender, scorecard, **kwargs) -> None:
            logger.info("episode %s scored: %s", scorecard.episode, scorecard.summary())

        def on_incident(sender, reason=None, **kwargs) -> None:
            logger.warning("integrity incident: %s", reason)

        def on_aggregation(sender, result=None, **kwargs) -> None:
            if result is not None:
                logger.info("aggregation: %s", result.report.describe())

        signals.episode_scored.connect(on_episode_scored, weak=False)
        signals.integrity_incident.connect(on_incident, weak=False)
        signals.aggregation_completed.connect(on_aggregation, weak=False)
