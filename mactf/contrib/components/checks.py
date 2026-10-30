"""The seven component checks (document 17 section 17.3.2).

A message id is the code of the table in ``mactf.components.codes`` namespaced by the app:
``mactf.components.E007`` is ``E007`` found by a check instead of raised mid-build. A check
never invents a code, and every message carries a hint that says what to do about it.
"""

from __future__ import annotations

from typing import Any

from ...checks import ERROR, WARNING, CheckMessage
from ...components import HUB, KIND_SPECS, ComponentResolver, ComponentSpec, all_kind_ids
from ...components.known import BUILTIN_NAMES, configured_specs, import_builtins
from ...exceptions import ComponentError, ComponentMissing, ImproperlyConfigured

_APP = "mactf.components"


def _error(code: str, message: str, hint: str | None = None) -> CheckMessage:
    return CheckMessage(level=ERROR, message=message, hint=hint, id=f"{_APP}.{code}")


def _warning(code: str, message: str, hint: str | None = None) -> CheckMessage:
    return CheckMessage(level=WARNING, message=message, hint=hint, id=f"{_APP}.{code}")


def _loaded() -> Any:
    """The settings, with the project's components imported (idempotent)."""
    from ...components.plugins import ensure_components_loaded
    from ...conf import get_settings

    settings = get_settings()
    ensure_components_loaded(settings)
    return settings


def check_registry_contracts() -> list[CheckMessage]:
    """``E007``/``E002``: every registered component meets its kind's contract."""
    _loaded()
    import_builtins()
    found: list[CheckMessage] = []
    for kind_id in all_kind_ids():
        components = HUB.registry(kind_id).components()
        folded: dict[str, str] = {}
        for component in components:
            problems = KIND_SPECS[kind_id].contract(component.factory)
            if problems:
                found.append(_error(
                    "E007",
                    f"{kind_id}:{component.name} breaks its contract: " + "; ".join(problems),
                    hint=f"Fix the class registered by {component.module or '?'}, or unregister it.",
                ))
            other = folded.setdefault(component.name.lower(), component.name)
            if other != component.name:
                found.append(_error(
                    "E002",
                    f"{kind_id}: {other!r} and {component.name!r} differ only by case",
                    hint="A YAML author cannot tell them apart; rename one.",
                ))
    return found


def check_configured_specs() -> list[CheckMessage]:
    """Every configured component resolves, with the same errors a build would raise."""
    settings = _loaded()
    import_builtins()
    resolver = ComponentResolver()
    found: list[CheckMessage] = []
    for slot, kind_id, raw in configured_specs(settings):
        spec = None
        try:
            spec = ComponentSpec.parse(kind_id, raw)
            # Settings are a trusted surface (15.3.2): ``import:`` there is the operator's own.
            resolver.validate(spec, trusted=True)
        except ComponentMissing as exc:
            if spec is not None and spec.type in BUILTIN_NAMES.get(kind_id, ()):
                continue                       # shipped, only not imported in this process
            found.append(_error(
                exc.code, f"{slot}: {exc}",
                hint=f"Accepted: {list(exc.known)}. A plugin's name needs its module in COMPONENTS.MODULES.",
            ))
        except ComponentError as exc:
            found.append(_error(
                exc.code, f"{slot}: {exc}",
                hint="Run `mactf components validate` with the same spec for the full contract.",
            ))
        except (ImportError, ImproperlyConfigured) as exc:
            found.append(_error(
                "E008", f"{slot}: {type(exc).__name__}: {exc}",
                hint="The module named by the spec could not be imported.",
            ))
    return found


def check_builtin_replacements() -> list[CheckMessage]:
    """``W001``: a built-in is replaced while ``COMPONENTS.ALLOW_BUILTIN_OVERRIDE`` is on."""
    settings = _loaded()
    if not (settings.get("COMPONENTS") or {}).get("ALLOW_BUILTIN_OVERRIDE"):
        return []
    found = []
    for kind_id in all_kind_ids():
        for component in HUB.registry(kind_id).components():
            if component.overrides_builtin:
                found.append(_warning(
                    "W001",
                    f"{kind_id}:{component.name} replaces a built-in (origin={component.origin.value})",
                    hint="A built-in is the framework's promise about a number; "
                         "turn COMPONENTS.ALLOW_BUILTIN_OVERRIDE off once the replacement is final.",
                ))
    return found


def check_dead_trainer_settings() -> list[CheckMessage]:
    """``W002``: a setting that looks like a selector and is read by nothing (2.2)."""
    from ...conf import DEFAULTS, get_settings
    from ...orchestration.dispatch import engine_name

    settings = get_settings()
    try:
        if engine_name(settings) != "orchestrator":
            return []
    except ImproperlyConfigured:
        return []                              # check_engine_support reports E090
    trainer = settings.get("TRAINER") or {}
    defaults = DEFAULTS["TRAINER"]
    return [
        _warning(
            "W002",
            f"TRAINER.{key} is set to {trainer[key]!r} but ORCHESTRATOR.ENGINE=orchestrator never reads it",
            hint="Set ORCHESTRATOR.ENGINE=trainer (13.5), or remove the setting.",
        )
        for key in ("PEER_STRATEGY", "WEIGHT_UPDATE")
        if trainer.get(key) not in (None, defaults.get(key))
    ]


def check_guard_modes() -> list[CheckMessage]:
    """``W003``/``W013``: a guard or a probe that is configured to look away."""
    from ...conf import get_settings

    settings = get_settings()
    found = []
    if str((settings.get("COMPLIANCE") or {}).get("GUARD_MODE", "enforce")).lower() == "fallback":
        found.append(_warning(
            "W003",
            "COMPLIANCE.GUARD_MODE is 'fallback': a scoring stage that breaks the floor "
            "is replaced by the reference stage instead of failing the run",
            hint="Use 'enforce' for a real run; 'fallback' hides a plugin bug behind a substituted score.",
        ))
    if str((settings.get("RL") or {}).get("VERIFY", "custom")).lower() == "off":
        found.append(_warning(
            "W013",
            "RL.VERIFY is 'off': custom RL functions run without a contract probe",
            hint="Set RL.VERIFY to 'custom' (the default) or 'all'.",
        ))
    return found


def check_unsafe_policy() -> list[CheckMessage]:
    """``W004``/``E009``: a policy that declares ``unsafe = True`` is configured (7.7)."""
    settings = _loaded()
    import_builtins()
    trainer = settings.get("TRAINER") or {}
    raw = trainer.get("POLICY")
    if raw is None:
        return []
    try:
        component = ComponentResolver().validate(ComponentSpec.parse("trainer_policy", raw), trusted=True)
    except (ComponentError, ImportError, ImproperlyConfigured):
        return []                              # check_configured_specs reports it
    if not getattr(component.factory, "unsafe", False):
        return []
    found = [_warning(
        "W004",
        f"trainer_policy:{component.name} is declared unsafe; StepGuard does not wrap it",
        hint="An unsafe policy manages its own safety; prefer a guarded one.",
    )]
    if not trainer.get("ALLOW_UNSAFE_POLICY", False):
        found.append(_error(
            "E009",
            f"trainer_policy:{component.name} is declared unsafe and TRAINER.ALLOW_UNSAFE_POLICY is off",
            hint="Set TRAINER.ALLOW_UNSAFE_POLICY=true to run it, or pick a guarded policy.",
        ))
    return found


def check_engine_support() -> list[CheckMessage]:
    """``E090``/``E092``: the engine named exists and can serve this installation."""
    from ...conf import get_settings
    from ...orchestration.dispatch import ENGINE_TRAINER, engine_name

    settings = get_settings()
    try:
        name = engine_name(settings)
    except ImproperlyConfigured as exc:
        return [_error("E090", str(exc).removeprefix("E090: "),
                       hint="Set ORCHESTRATOR.ENGINE to 'orchestrator' or 'trainer'.")]
    if name == ENGINE_TRAINER and any(
        str(app).startswith("mactf.distributed") for app in settings.get("INSTALLED_APPS") or ()
    ):
        return [_error(
            "E092",
            "ORCHESTRATOR.ENGINE='trainer' cannot run with mactf.distributed installed",
            hint="Use ORCHESTRATOR.ENGINE='orchestrator' for a distributed run.",
        )]
    return []
