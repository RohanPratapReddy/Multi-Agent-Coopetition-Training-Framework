"""System checks registered by the coopetition app."""

from __future__ import annotations

from ... import checks
from .apps import check_aggregation_bounds

checks.register(check_aggregation_bounds, app="mactf_coopetition")
