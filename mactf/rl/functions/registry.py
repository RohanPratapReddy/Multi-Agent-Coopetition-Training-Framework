"""The function spec registry (document 07 section 10).

One registry per family. Specs are ``{"type": <key>, **kwargs}`` with no silent
fallback: a typo must never train the wrong objective.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...exceptions import ValidationError
from ._spec import build_from_kwargs
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

#: Family name -> ``spec["type"]`` -> class.
FUNCTION_TYPES: dict[str, dict[str, type]] = {
    "value": VALUE_FUNCTION_TYPES,
    "advantage": ADVANTAGE_FUNCTION_TYPES,
    "loss": LOSS_FUNCTION_TYPES,
    "rollup": {
        "kappa": KappaWeightedRollup,
        "mean": MeanRollup,
        "sum": SumRollup,
        "minimum": MinimumRollup,
        "worst_case": WorstCaseRollup,
        "explicit": ExplicitRollup,
    },
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
CAPABILITY_ORDER: tuple[str, ...] = (
    "value",
    "advantage",
    "loss",
    "rollup",
    "return",
    "reward",
    "regularization",
    "objective",
    "gradient",
    "metric",
    "action",
    "synthesis",
)


def register_function(family: str, name: str, cls: type, *, replace: bool = False) -> None:
    """Add a wrapper type to a family. Applications use this for custom wrappers."""
    if family not in FUNCTION_TYPES:
        raise ValidationError(
            f"unknown function family {family!r}; accepted families: "
            + ", ".join(sorted(FUNCTION_TYPES))
        )
    if not isinstance(cls, type):
        raise ValidationError(f"{cls!r} is not a class")
    if not name or not name.replace("_", "").isalnum():
        raise ValidationError(f"invalid function type name {name!r}")
    table = FUNCTION_TYPES[family]
    if name in table and not replace:
        raise ValidationError(
            f"{family}.{name} is already registered to {table[name].__name__}; "
            "pass replace=True to override"
        )
    table[name] = cls


def resolve_family(type_name: str) -> str:
    """Find the family that owns ``type_name``, for ``build_function(spec)``.

    A name that appears in more than one family (``callable`` is in six) is
    ambiguous and is rejected rather than guessed: the caller must pass
    ``family=``.
    """
    matches = [family for family, table in FUNCTION_TYPES.items() if type_name in table]
    if not matches:
        raise ValidationError(
            f"unknown function type {type_name!r}; accepted types: "
            + ", ".join(sorted({name for table in FUNCTION_TYPES.values() for name in table}))
        )
    if len(matches) > 1:
        raise ValidationError(
            f"function type {type_name!r} is ambiguous between families "
            + ", ".join(sorted(matches))
            + "; pass family= to disambiguate"
        )
    return matches[0]


def build_function(spec: Mapping[str, Any], *, family: str | None = None) -> Any:
    """Construct a wrapper from ``{"type": <key>, **kwargs}``."""
    payload = dict(spec or {})
    type_name = payload.pop("type", None)
    if type_name is None:
        raise ValidationError(
            "a function spec requires a 'type' key; families: "
            + ", ".join(sorted(FUNCTION_TYPES))
        )
    if family is None:
        family = resolve_family(str(type_name))
    elif family not in FUNCTION_TYPES:
        raise ValidationError(
            f"unknown function family {family!r}; accepted families: "
            + ", ".join(sorted(FUNCTION_TYPES))
        )
    table = FUNCTION_TYPES[family]
    if type_name not in table:
        raise ValidationError(
            f"unknown {family} type {type_name!r}; accepted types: "
            + ", ".join(sorted(table))
        )
    return build_from_kwargs(table[type_name], payload)


def function_capabilities() -> dict[str, list[dict[str, Any]]]:
    """The catalog backing ``GET /trainers/capabilities``."""
    payload: dict[str, list[dict[str, Any]]] = {}
    for family in CAPABILITY_ORDER:
        table = FUNCTION_TYPES[family]
        entries: list[dict[str, Any]] = []
        for name in sorted(table):
            cls = table[name]
            entry: dict[str, Any] = {
                "type": name,
                "class": cls.__name__,
            }
            required = getattr(cls, "required_fields", None)
            if required:
                entry["required_fields"] = sorted(required)
            from ._spec import constructor_keys

            entry["kwargs"] = sorted(constructor_keys(cls))
            entries.append(entry)
        payload[family] = entries
    return payload


__all__ = [
    "ACTION_TYPES",
    "ADVANTAGE_TYPES",
    "CAPABILITY_ORDER",
    "FUNCTION_TYPES",
    "GRADIENT_TYPES",
    "LOSS_TYPES",
    "METRIC_TYPES",
    "OBJECTIVE_TYPES",
    "REGULARIZATION_TYPES",
    "RETURN_TYPES",
    "REWARD_TYPES",
    "ROLLUP_TYPES",
    "SYNTHESIS_TYPES",
    "VALUE_TYPES",
    "build_function",
    "function_capabilities",
    "register_function",
    "resolve_family",
]
