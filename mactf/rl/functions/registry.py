"""The function spec registry (document 07 section 10, document 17 section 8.2).

One registry per family, now backed by ``mactf.components``. Specs are
``{"type": <key>, **kwargs}`` (flat, as before) or ``{"type": ..., "params": {...}}`` or
``{"import": "pkg.mod:Class", ...}``. There is still no silent fallback: a typo must never
train the wrong objective.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...components import HUB
from ...components.decorators import register_function, unregister
from ...components.kinds import RL_FAMILIES, ComponentKind
from ...components.legacy import LegacyTable
from ...components.resolver import ComponentResolver, ComponentSpec
from ...exceptions import ValidationError
from .advantage import ADVANTAGE_FUNCTION_TYPES
from .loss import LOSS_FUNCTION_TYPES
from .rollup import (
    ExplicitRollup,
    KappaWeightedRollup,
    MeanRollup,
    MinimumRollup,
    SumRollup,
    WorstCaseRollup,
)
from .support import SUPPORT_FAMILIES
from .value import VALUE_FUNCTION_TYPES

#: Names that are second spellings of another entry. ``function_capabilities`` reports them
#: with ``alias_of`` so a UI can fold them. ``kappa_weighted`` is the name
#: ``scoring/hierarchy.py:97-104`` always accepted and ``HierarchicalScorer`` defaults to.
ALIASES: dict[tuple[str, str], str] = {("rollup", "kappa_weighted"): "kappa"}

#: Family name -> ``spec["type"]`` -> class. Each value is a live view of its registry.
FUNCTION_TYPES: dict[str, dict[str, type]] = {
    "value": VALUE_FUNCTION_TYPES,
    "advantage": ADVANTAGE_FUNCTION_TYPES,
    "loss": LOSS_FUNCTION_TYPES,
    "rollup": LegacyTable(
        "rl_function.rollup",
        {
            "kappa": KappaWeightedRollup,
            "kappa_weighted": KappaWeightedRollup,
            "mean": MeanRollup,
            "sum": SumRollup,
            "minimum": MinimumRollup,
            "worst_case": WorstCaseRollup,
            "explicit": ExplicitRollup,
        },
    ),
    **SUPPORT_FAMILIES,
}

#: Convenience aliases so ``from ...registry import LOSS_TYPES`` reads well.
VALUE_TYPES = VALUE_FUNCTION_TYPES
ADVANTAGE_TYPES = ADVANTAGE_FUNCTION_TYPES
LOSS_TYPES = LOSS_FUNCTION_TYPES
ROLLUP_TYPES = FUNCTION_TYPES["rollup"]
RETURN_TYPES = FUNCTION_TYPES["return"]
REWARD_TYPES = FUNCTION_TYPES["reward"]
REGULARIZATION_TYPES = FUNCTION_TYPES["regularization"]
OBJECTIVE_TYPES = FUNCTION_TYPES["objective"]
GRADIENT_TYPES = FUNCTION_TYPES["gradient"]
METRIC_TYPES = FUNCTION_TYPES["metric"]
ACTION_TYPES = FUNCTION_TYPES["action"]
SYNTHESIS_TYPES = FUNCTION_TYPES["synthesis"]

#: Families exposed by ``function_capabilities()``; the tuple fixes the order.
CAPABILITY_ORDER: tuple[str, ...] = RL_FAMILIES

_RESOLVER = ComponentResolver()


def _kind_id(family: str) -> str:
    """``rl_function.<family>``; raises the historical message for an unknown family."""
    if family not in FUNCTION_TYPES:
        raise ValidationError(
            f"unknown function family {family!r}; accepted families: "
            + ", ".join(sorted(FUNCTION_TYPES))
        )
    return ComponentKind.RL_FUNCTION.kind_id(family)


def unregister_function(family: str, name: str, *, scope: str = "global") -> None:
    """Remove a custom function. Built-ins are refused (``ComponentLocked``, section 4.3)."""
    unregister(ComponentKind.RL_FUNCTION, name, family=family, scope=scope)


def resolve_family(type_name: str) -> str:
    """Find the family that owns ``type_name``, for ``build_function(spec)``.

    A name that appears in more than one family (``callable`` is in six) is
    ambiguous and is rejected rather than guessed: the caller must pass
    ``family=``. The search sees run-scoped overlays, so a function registered with
    ``scope="run"`` resolves inside that run only.
    """
    matches = [
        family for family in CAPABILITY_ORDER
        if HUB.registry(_kind_id(family)).has(type_name)
    ]
    if not matches:
        known = sorted({
            name for family in CAPABILITY_ORDER for name in HUB.registry(_kind_id(family)).names()
        })
        raise ValidationError(
            f"unknown function type {type_name!r}; accepted types: " + ", ".join(known)
        )
    if len(matches) > 1:
        raise ValidationError(
            f"function type {type_name!r} is ambiguous between families "
            + ", ".join(sorted(matches))
            + "; pass family= to disambiguate"
        )
    return matches[0]


def build_function(
    spec: Mapping[str, Any], *, family: str | None = None, inject: Mapping[str, Any] | None = None,
) -> Any:
    """Construct a wrapper from ``{"type": <key>, **kwargs}`` (or the ``params`` / ``import`` forms).

    Every error ``ValidationError`` raised before is raised with the same text; the
    component errors it can additionally raise subclass ``ValidationError`` (17.1).
    """
    payload = dict(spec or {})
    type_name = payload.get("type")
    import_path = payload.get("import")
    if type_name is None and import_path is None:
        raise ValidationError(
            "a function spec requires a 'type' key; families: " + ", ".join(sorted(FUNCTION_TYPES))
        )
    if family is None:
        if type_name is None:
            raise ValidationError(
                "an 'import' spec cannot name its own family; pass family= to build_function"
            )
        family = resolve_family(str(type_name))
    kind_id = _kind_id(family)
    registry = HUB.registry(kind_id)
    if import_path is None and not registry.has(str(type_name)):
        raise ValidationError(
            f"unknown {family} type {type_name!r}; accepted types: " + ", ".join(registry.names())
        )
    parsed = ComponentSpec.parse(kind_id, payload)
    # The legacy contract: a wrapper is checked when it is *called* (and, for a custom one,
    # by the contract probe), not when it is named or built; see ``ComponentRegistry.build``.
    return _RESOLVER.resolve(parsed, trusted=True, inject=inject, check_instance=False).instance


def function_capabilities() -> dict[str, list[dict[str, Any]]]:
    """The catalog backing ``GET /trainers/capabilities`` and ``GET /components``.

    The keys ``type``, ``class``, ``kwargs`` and ``required_fields`` are unchanged; each
    entry gains ``origin``, ``version``, ``mutability`` and, for an alias, ``alias_of``.
    """
    payload: dict[str, list[dict[str, Any]]] = {}
    for family in CAPABILITY_ORDER:
        entries: list[dict[str, Any]] = []
        for component in HUB.registry(_kind_id(family)).components():
            cls = component.factory
            entry: dict[str, Any] = {
                "type": component.name,
                "class": getattr(cls, "__name__", component.qualname),
                "origin": component.origin.value,
                "version": component.version,
                "mutability": component.mutability.value,
            }
            required = getattr(cls, "required_fields", None)
            if required:
                entry["required_fields"] = sorted(required)
            entry["kwargs"] = sorted(component.accepted_params())
            alias = ALIASES.get((family, component.name))
            if alias is not None:
                entry["alias_of"] = alias
            entries.append(entry)
        payload[family] = entries
    return payload


__all__ = [
    "ACTION_TYPES", "ADVANTAGE_TYPES", "ALIASES", "CAPABILITY_ORDER", "FUNCTION_TYPES",
    "GRADIENT_TYPES", "LOSS_TYPES", "METRIC_TYPES", "OBJECTIVE_TYPES", "REGULARIZATION_TYPES",
    "RETURN_TYPES", "REWARD_TYPES", "ROLLUP_TYPES", "SYNTHESIS_TYPES", "VALUE_TYPES",
    "build_function", "function_capabilities", "register_function", "resolve_family",
    "unregister_function",
]
