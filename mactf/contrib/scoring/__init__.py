"""MACTF scoring app."""

from __future__ import annotations

from .apps import ScoringConfig

default_app_config = "mactf.contrib.scoring.apps.ScoringConfig"

__all__ = ["ScoringConfig", "default_app_config"]
