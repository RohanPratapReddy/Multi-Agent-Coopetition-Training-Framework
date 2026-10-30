"""System checks registered by the components app."""

from __future__ import annotations

from ... import checks
from .checks import (
    check_builtin_replacements,
    check_configured_specs,
    check_dead_trainer_settings,
    check_engine_support,
    check_guard_modes,
    check_registry_contracts,
    check_unsafe_policy,
)

for _check in (
    check_registry_contracts, check_configured_specs, check_builtin_replacements,
    check_dead_trainer_settings, check_guard_modes, check_unsafe_policy, check_engine_support,
):
    checks.register(
        _check, app="mactf_components", requires="mactf.contrib.components", tags=("components",)
    )
