"""Loading components: entry points and ``COMPONENTS.MODULES`` (document 17 sections 4.2, 15.5).

Nothing is registered by being importable (invariant I5). Loading is a single idempotent
function. :func:`ensure_components_loaded` is what ``conf.configure`` and the engines call:
it compares the canonical ``COMPONENTS`` block with the last one loaded, so a second call
in the same process imports nothing and a changed block reloads.
"""

from __future__ import annotations

import importlib
import logging
import threading
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Mapping

from ..exceptions import ComponentBuildError, ComponentError, ImproperlyConfigured
from .kinds import Origin
from .reflect import canonical_json
from .registry import HUB, HubPolicy

log = logging.getLogger("mactf.components")

ENTRY_POINT_GROUP = "mactf.components"

_LOCK = threading.RLock()
_LOADED: dict[str, Origin] = {}
_LAST_BLOCK: dict[str, Any] | None = None
_LAST_REPORT: "LoadReport | None" = None


@dataclass
class LoadReport:
    plugins: list[str] = field(default_factory=list)
    modules: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    policy: HubPolicy | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "plugins": list(self.plugins),
            "modules": list(self.modules),
            "skipped": list(self.skipped),
        }


def fold_block(components: Mapping[str, Any] | None) -> dict[str, Any]:
    """Upper-case the keys of a ``COMPONENTS`` block and fold the ``API_ALLOW_IMPORT`` alias.

    ``conf`` validates a block this way, so the hub reads it this way too: a key a validator
    accepted is never ignored here. The canonical spelling wins when both are given (``conf``
    already refused the case where they disagree).
    """
    folded: dict[str, Any] = {}
    for key, value in dict(components or {}).items():
        name = str(key).upper()
        if name == "API_ALLOW_IMPORT":
            folded.setdefault("ALLOW_IMPORT", value)
        else:
            folded[name] = value
    return folded


def policy_from_settings(components: Mapping[str, Any], *, locked: bool = True) -> HubPolicy:
    """Build a :class:`HubPolicy` from the ``COMPONENTS`` block (defaults in section 15.1)."""
    block = fold_block(components)
    return HubPolicy(
        allow_builtin_override=bool(block.get("ALLOW_BUILTIN_OVERRIDE", False)),
        locked=frozenset(str(p) for p in block.get("LOCKED", ())) if locked else frozenset(),
        strict_legacy=bool(block.get("STRICT_LEGACY", False)),
        allow_import=bool(block.get("ALLOW_IMPORT", False)),
        import_allowlist=tuple(str(p) for p in block.get("IMPORT_ALLOWLIST", ())),
        max_components_per_kind=int(block.get("MAX_PER_KIND", 512)),
        strict_restore=bool(block.get("STRICT_RESTORE", False)),
    )


def _import(name: str, origin: Origin) -> None:
    with HUB.loading(origin):
        try:
            importlib.import_module(name)
        except ComponentError:
            raise
        except ImportError as error:
            raise ImproperlyConfigured(
                f"COMPONENTS could not import module {name!r}: {error}"
            ) from error
        except Exception as error:  # noqa: BLE001 - a module that raises is E008, cause chained
            raise ComponentBuildError("module", name, error) from error


def _load_entry_point(entry: metadata.EntryPoint) -> None:
    with HUB.loading(Origin.PLUGIN):
        try:
            target = entry.load()
            if callable(target):
                # ``entry = "pkg.module:install"``: ``install(hub)`` performs the registrations.
                target(HUB)
        except ComponentError:
            raise
        except Exception as error:  # noqa: BLE001 - chained, never swallowed
            raise ComponentBuildError("plugin", entry.name, error) from error


def _plugin_names(setting: Any) -> set[str] | None:
    """``None`` loads every entry point; an empty set loads none."""
    if isinstance(setting, (list, tuple)):
        return {str(name) for name in setting}
    text = str(setting).lower()
    if text == "auto":
        return None
    if text == "off":
        return set()
    raise ImproperlyConfigured(
        "E015: COMPONENTS.PLUGINS must be 'auto', 'off' or a list of entry-point names"
    )


def load_components(components: Mapping[str, Any]) -> LoadReport:
    """Two-phase, idempotent load.

    Phase 1 applies the policy *without* locks so plugins and project modules can
    register. Phase 2 applies the full policy (locks) and optionally freezes the
    registries. Calling it again with the same settings imports nothing twice. If a plugin
    or module fails, the previous policy is put back and the error propagates.
    """
    report = LoadReport()
    components = fold_block(components)
    with _LOCK:
        previous = HUB.policy
        names = _plugin_names(components.get("PLUGINS", "auto"))
        # A reload with a changed block starts from open registries; a freeze belongs to
        # the configuration that asked for it.
        HUB.unfreeze_all()
        HUB.configure(policy_from_settings(components, locked=False))
        try:
            for entry in metadata.entry_points(group=ENTRY_POINT_GROUP):
                if names is not None and entry.name not in names:
                    report.skipped.append(entry.name)
                    continue
                key = f"plugin:{entry.name}={entry.value}"
                if key not in _LOADED:
                    _load_entry_point(entry)
                    _LOADED[key] = Origin.PLUGIN
                report.plugins.append(entry.name)

            for module in components.get("MODULES", ()):
                name = str(module)
                if name not in _LOADED:
                    _import(name, Origin.PROJECT)
                    _LOADED[name] = Origin.PROJECT
                report.modules.append(name)
        except BaseException:
            HUB.configure(previous)
            raise

        final = policy_from_settings(components, locked=True)
        HUB.configure(final)
        report.policy = final
        if bool(components.get("FREEZE_AFTER_LOAD", False)):
            HUB.freeze_all()
    log.info("components loaded: %s", report.as_dict())
    return report


def ensure_components_loaded(settings: Any = None) -> LoadReport:
    """Load components once per distinct ``COMPONENTS`` block; safe to call anywhere.

    Every caller needs the same guarantee and none can be sure it is first: ``conf.configure``
    (the CLI), ``TrainerEngine.prepare`` (a run built in code), ``RunManager.create`` (the API
    runs several), ``mactf check``. Comparing the canonical block means a settings reload in a
    test loads the new one, and a second call in the same process does nothing.
    """
    global _LAST_BLOCK, _LAST_REPORT
    if settings is None:
        from ..conf import get_settings

        settings = get_settings()
    block = dict(settings.get("COMPONENTS") or {}) if hasattr(settings, "get") else {}
    fingerprint = canonical_json(block)
    with _LOCK:
        if _LAST_BLOCK is not None and canonical_json(_LAST_BLOCK) == fingerprint:
            assert _LAST_REPORT is not None
            return _LAST_REPORT
        report = load_components(block)
        _LAST_BLOCK = block
        _LAST_REPORT = report
        return report


def reset_for_tests() -> None:
    """Forget what was loaded. Test-only: registrations themselves are not undone."""
    global _LAST_BLOCK, _LAST_REPORT
    with _LOCK:
        _LOADED.clear()
        _LAST_BLOCK = None
        _LAST_REPORT = None
        HUB.unfreeze_all()
        HUB.configure(HubPolicy())
