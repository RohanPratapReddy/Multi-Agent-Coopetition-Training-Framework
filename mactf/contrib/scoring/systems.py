"""System checks registered by the scoring app."""

from __future__ import annotations

from ... import checks
from .apps import check_rule_catalog, check_settings_consistency

checks.register(check_rule_catalog, app="mactf_scoring")
checks.register(check_settings_consistency, app="mactf_scoring")
