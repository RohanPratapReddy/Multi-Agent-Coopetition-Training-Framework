"""The built-in component names, known without importing torch or populating a registry.

``conf.py`` validates a component reference at settings load, on every command, on a path
that must stay torch-free. The registries cannot answer there: they are populated by
modules that import torch. So the built-ins are listed here, once, and
``tests/components/test_known.py`` keeps this table equal to the live legacy tables.

A kind absent from ``BUILTIN_NAMES`` is not name-checked at load: its reference is validated
by shape, and the registry decides at build with a precise ``ComponentMissing``. That is the
right split for kinds whose names are *derived* (every ``torch.optim`` class is a valid
``OPTIMIZER.CLASS``; every scheduler factory is a registry entry resolved later). A phase
that adds built-ins of a kind (``reward_term``, ``trainer_policy``, ``aggregator``) adds the
kind's row here in the same change.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from .kinds import RL_FAMILIES, all_kind_ids

#: ``kind_id -> names the framework ships``. A kind that is not here is shape-checked only.
BUILTIN_NAMES: dict[str, frozenset[str]] = {
    "scorer": frozenset(
        {"reward", "penalty", "warning", "hierarchy", "composed_reward", "zero_reward", "no_warnings"}
    ),
    "reward_term": frozenset({"severity_scaled", "capped"}),
    "peer_strategy": frozenset(
        {"independent", "cooperative", "kappa_softmax", "frozen_peer", "hierarchical"}
    ),
    "weight_update": frozenset(
        {"kappa_softmax", "uniform", "score_weighted", "reference_only", "none", "explicit"}
    ),
    "aggregator": frozenset({"default"}),
    "trainer_policy": frozenset({"default", "co_distill"}),
}

#: The codes document 17 section 14 introduces, with the one-line meaning P6's code table prints.
#: ``tests/components/test_known.py`` keeps every code the restorer can emit in this table.
PERSISTENCE_CODES: dict[str, str] = {
    "E093": "a checkpoint needs a component this build has not registered",
    "E094": "a live slot differs from the recorded entry and COMPONENTS.STRICT_RESTORE is on",
    "E095": "the checkpoint's components section, or one entry, cannot be interpreted",
    "E096": "an entry's component schema is newer than this build understands",
    "W052": "a slot differs from the recorded entry (or names a slot the run no longer has)",
    "W053": "the checkpoint records no components; nothing was verified",
}

#: Every kind id a ``COMPONENTS.LOCKED`` pattern may name, ``rl_function.<family>`` included.
LOCKABLE_KINDS: frozenset[str] = frozenset(all_kind_ids())

#: Kinds whose names are lower-cased before comparison, because a YAML author writes
#: ``Kappa_Softmax`` and the registry key is lower case (the registry enforces the same).
LOWERCASE_KINDS: frozenset[str] = frozenset({"peer_strategy", "weight_update"})

#: ``str.isidentifier`` per segment; the same rule ``conf_files`` uses for a dotted path.
_DOTTED = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
#: A lock pattern's prefix: a kind id, a family-qualified kind id, or ``slot``.
_LOCK = re.compile(r"^(slot|rl_function(\.[a-z_*?\[\]]+)?|[a-z_*?\[\]]+):(.+)$")


def is_dotted_name(text: Any) -> bool:
    """Whether ``text`` is a dotted Python name (``acme.components``), not a path."""
    return isinstance(text, str) and bool(_DOTTED.match(text))


def registry_names(kind_id: str) -> frozenset[str]:
    """The names currently registered, or nothing when the built-ins are not loaded yet.

    Reading must not import torch: a registry whose built-ins are not loaded has had no
    registration at all (registering loads them first), so the static table answers.
    """
    try:
        from .registry import HUB

        registry = HUB.registry(kind_id)
        if not registry.builtins_loaded:
            return frozenset()
        return frozenset(registry.names())
    except Exception:  # noqa: BLE001 - a bare unit test may have no hub; names are static
        return frozenset()


def known_names(kind_id: str) -> frozenset[str] | None:
    """Every name a project may write for ``kind_id``, or ``None`` when not name-checked.

    The union, not the registry alone: at load time the registry is empty, and after a
    plugin loads it is a superset of the built-ins. Both directions are correct answers to
    "could this name exist?" and neither can be false.
    """
    static = BUILTIN_NAMES.get(kind_id)
    if static is None:
        return None
    return static | registry_names(kind_id)


def may_be_extended(components: Mapping[str, Any] | None) -> bool:
    """Whether a name the framework does not ship could legitimately exist at build.

    Two ways: ``COMPONENTS.MODULES`` names something to import, or a distribution exposes
    the ``mactf.components`` entry-point group. Both checks are import-free.
    """
    block = {str(key).upper(): value for key, value in dict(components or {}).items()}
    if block.get("MODULES"):
        return True
    plugins = block.get("PLUGINS", "auto")
    if isinstance(plugins, (list, tuple)):
        if not plugins:
            return False
    elif str(plugins).lower() == "off":
        return False
    from importlib import metadata

    return bool(tuple(metadata.entry_points(group="mactf.components")))


def spec_shape_problem(raw: Any) -> str | None:
    """``None`` when ``raw`` can be a component spec. Names are not checked here."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return None if raw.strip() else "an empty string is not a component name"
    if isinstance(raw, Mapping):
        keys = {str(key) for key in raw}
        if not ({"type", "name", "import"} & keys):
            return "a spec mapping needs 'type' (or 'import')"
        if "params" in keys and not isinstance(raw.get("params"), Mapping):
            return "'params' must be a mapping"
        return None
    return f"a spec is a name or a mapping, not {type(raw).__name__}"


def component_ref_problem(where: str, raw: Any, *, kind_id: str, settings: Any) -> str | None:
    """Validate one component reference. ``None`` when it is acceptable *at load*.

    Three outcomes, and the third is the point of ``may_be_extended``:

    1. A name the framework ships: accepted.
    2. A mapping with ``type`` or ``import``: accepted by shape. The name inside ``type``
       is checked the same way when present; ``import`` is checked for shape only, because
       whether the module exists is a build-time fact (and an untrusted surface is refused
       there, by the resolver, not here).
    3. A name nothing ships: refused **unless** a project module or a plugin could add it,
       in which case it is accepted here and ``E001`` at build.
    """
    problem = spec_shape_problem(raw)
    if problem is not None:
        return f"{where}: {problem}"
    if isinstance(raw, Mapping):
        candidate = raw.get("type", raw.get("name"))
        if candidate is None:
            return None  # ``import`` only: shape is all a settings load can know
    else:
        candidate = raw
    text = str(candidate).strip()
    if not text:
        return f"{where}: an empty component name"
    names = known_names(kind_id)
    if names is None:
        return None
    folded = text.lower() if kind_id in LOWERCASE_KINDS else text
    if folded in names:
        return None
    components = dict(settings.get("COMPONENTS") or {}) if hasattr(settings, "get") else {}
    if may_be_extended(components):
        return None
    return f"{where} must be one of " + ", ".join(sorted(names))


def lock_pattern_problem(pattern: Any, *, kinds: Any) -> str | None:
    """One ``COMPONENTS.LOCKED`` entry, or why it cannot lock anything."""
    text = str(pattern)
    if not isinstance(pattern, str) or not text or text != text.strip():
        return f"COMPONENTS.LOCKED: {pattern!r} is not a usable pattern"
    match = _LOCK.match(text)
    if match is None:
        return (
            f"COMPONENTS.LOCKED: {pattern!r} must be 'kind:name' or 'slot:<path>' "
            "(for example 'scorer:penalty' or 'slot:TRAINER.WEIGHT_UPDATE')"
        )
    head = match.group(1)
    if head != "slot" and not _kind_matches(head, kinds):
        return (
            f"COMPONENTS.LOCKED: {pattern!r} names unknown kind {head!r}; kinds are "
            + ", ".join(sorted(kinds))
            + ", or 'slot'"
        )
    return None


def _kind_matches(head: str, kinds: Any) -> bool:
    """A kind head is a kind id, or an ``fnmatch`` pattern that matches at least one."""
    import fnmatch

    if head in kinds:
        return True
    if any(char in head for char in "*?["):
        return any(fnmatch.fnmatchcase(kind, head) for kind in kinds)
    return False


def import_builtins() -> None:
    """Import every module that registers framework components (imports torch).

    ``trainer_policy`` has no ``builtins_module`` (``mactf.training.policy`` needs torch and
    the torch-free import test registers a policy without it), so a command that *lists*
    components - ``mactf components list`` - calls this first.
    """
    import importlib

    from .registry import HUB

    for kind_id in all_kind_ids():
        HUB.registry(kind_id)._ensure_builtins()
    importlib.import_module("mactf.training.policy")


#: ``TRAINER`` key -> kind id, for a key that holds one spec.
_TRAINER_SPECS: dict[str, str] = {
    "PEER_STRATEGY": "peer_strategy",
    "WEIGHT_UPDATE": "weight_update",
    "POLICY": "trainer_policy",
    "AGGREGATOR": "aggregator",
}
#: ``TRAINER`` key -> kind id, for a key that maps ``glob -> spec``.
_TRAINER_OVERRIDES: dict[str, str] = {
    "PEER_STRATEGY_OVERRIDES": "peer_strategy",
    "WEIGHT_UPDATE_OVERRIDES": "weight_update",
}


def configured_specs(settings: Any) -> list[tuple[str, str, Any]]:
    """Every component spec the settings name, as ``(slot path, kind id, raw spec)``.

    Torch-free and registry-free: it only reads the settings. ``TRAINER.*``, ``RL.*`` and
    ``SCORING.STAGES.*`` are covered. ``OPTIMIZER`` is not listed: its ``TYPE`` is a backend
    word (``torch``, ``scipy``, ``callable``) or a torch class name as often as a registered
    name, and the optimizer plan validates it at build with the right knowledge.
    """
    found: list[tuple[str, str, Any]] = []
    get = settings.get if hasattr(settings, "get") else (lambda key, default=None: default)
    trainer = get("TRAINER") or {}
    for key, kind_id in _TRAINER_SPECS.items():
        raw = trainer.get(key)
        if raw is not None:
            found.append((f"TRAINER.{key}", kind_id, raw))
    for key, kind_id in _TRAINER_OVERRIDES.items():
        table = trainer.get(key) or {}
        if isinstance(table, Mapping):
            for glob, raw in table.items():
                found.append((f"TRAINER.{key}.{glob}", kind_id, raw))
    rl = get("RL") or {}
    families = {name.upper(): name for name in RL_FAMILIES}
    if isinstance(rl, Mapping):
        for key, raw in rl.items():
            family = families.get(str(key).upper())
            if family is not None and raw is not None:
                found.append((f"RL.{key}", f"rl_function.{family}", raw))
        overrides = rl.get("OVERRIDES") or {}
        if isinstance(overrides, Mapping):
            for key, table in overrides.items():
                family = families.get(str(key).upper())
                if family is not None and isinstance(table, Mapping):
                    for glob, raw in table.items():
                        found.append((f"RL.OVERRIDES.{key}.{glob}", f"rl_function.{family}", raw))
    stages = (get("SCORING") or {}).get("STAGES") or {}
    if isinstance(stages, Mapping):
        for kind, raw in stages.items():
            if raw is not None:
                found.append((f"SCORING.STAGES.{kind}", "scorer", raw))
    return found
