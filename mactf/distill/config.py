"""Settings defaults, key sets, validation, and the resolved run configuration.

This module is deliberately free of ``torch`` and of every other ``mactf.distill``
module. ``mactf.conf`` imports it to install ``DEFAULTS["DISTILLATION"]`` and to run
``validate_settings``, and ``conf`` is imported by nearly everything, including the
client package that must stay light. One validator (:func:`validate_block`) serves
both the settings path and the ``distillation=`` argument path, so the two can never
disagree about what a legal block is.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from ..exceptions import ImproperlyConfigured, ValidationError

__all__ = [
    "CONFORMANCE_KEYS",
    "CONFORMANCE_MODES",
    "DISTILLATION_DEFAULTS",
    "DISTILLATION_KEYS",
    "MAX_TEACHERS",
    "SIGMA_MODE_NAMES",
    "VIEW_KIND_NAMES",
    "DistillationConfig",
    "validate_block",
]

MAX_TEACHERS = 64

CONFORMANCE_MODES: frozenset[str] = frozenset({"declared", "audit", "hybrid"})

#: Every key ``DISTILLATION.CONFORMANCE`` accepts. An unknown key is an error rather
#: than an ignored typo: ``TRUST_DECAY`` misspelt is a gate silently running at its
#: default.
CONFORMANCE_KEYS: frozenset[str] = frozenset(
    {
        "ENABLED",
        "MODE",
        "TRUST_INIT",
        "TRUST_FLOOR",
        "TRUST_DECAY",
        "TRUST_RESTORE",
        "TRUST_MIN",
        "STRICT_UNDECLARED",
        "AUDIT_ARTIFACTS",
        "SIGMA_TRUST_MIN",
    }
)

SIGMA_MODE_NAMES: frozenset[str] = frozenset({"uniform", "teacher_softmax", "quality_softmax"})
VIEW_KIND_NAMES: frozenset[str] = frozenset(
    {"identity", "linear", "fixed_random", "hidden", "distribution"}
)

DISTILLATION_DEFAULTS: dict[str, Any] = {
    "ENABLED": False,
    "TEACHERS": [],
    "OUT_DIM": 64,
    "TAU": 0.07,
    "TAU_MIN": 0.01,
    "TAU_MAX": 1.0,
    "SYMMETRIC": True,
    "BETA": 1.0,
    "SIGMA_MODE": "quality_softmax",
    "SIGMA_FLOOR": 0.0,
    "SIGMA_CAP": 1.0,
    "SIGMA_DECAY": 0.9,
    "LAMBDA_COOP": 0.0,
    "LAMBDA_IND": 1.0,
    "LAMBDA_REG": 0.0,
    "LAMBDA_CD": 1.0,
    "PEER_MARGIN": 0.25,
    "WARMUP_STEPS": 0,
    "RAMP_STEPS": 0,
    "REGULARIZERS": [],
    "VIEWS": {},
    "STRICT": True,
    "CONFORMANCE": {
        "ENABLED": True,
        "MODE": "hybrid",
        "TRUST_INIT": 1.0,
        "TRUST_FLOOR": 0.0,
        "TRUST_DECAY": 0.5,
        "TRUST_RESTORE": 0.05,
        "TRUST_MIN": 0.05,
        "STRICT_UNDECLARED": True,
        "AUDIT_ARTIFACTS": {},
        "SIGMA_TRUST_MIN": True,
    },
}

#: Every key the block accepts. Derived from the defaults so a key added to one
#: cannot be forgotten in the other.
DISTILLATION_KEYS: frozenset[str] = frozenset(DISTILLATION_DEFAULTS)


def _teacher_name(spec: Any) -> str | None:
    if isinstance(spec, Mapping):
        name = spec.get("name") or spec.get("id")
        return None if name is None else str(name)
    name = getattr(spec, "name", None)
    return None if name is None else str(name)


def _as_float(block: Mapping[str, Any], key: str, default: float, owner: str) -> float:
    value = block.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ImproperlyConfigured(f"{owner}{key} must be a number, got {value!r}") from error


def validate_block(block: Mapping[str, Any], *, require_teachers: bool | None = None) -> None:
    """Validate one ``DISTILLATION`` mapping (document 14 section 10.4).

    ``require_teachers`` defaults to the block's own ``ENABLED``. The caller that
    has resolved the toggle elsewhere (``DistillationConfig.from_spec``) passes the
    resolved value, because a ``True`` argument over a ``ENABLED: false`` block is
    enabled whatever the block says.

    Raises ``ImproperlyConfigured`` and names the key and the accepted values in
    every message.
    """
    if not isinstance(block, Mapping):
        raise ImproperlyConfigured("DISTILLATION must be a mapping")
    owner = "DISTILLATION."
    unknown = sorted(set(block) - DISTILLATION_KEYS)
    if unknown:
        raise ValidationError(
            "DISTILLATION does not accept "
            + ", ".join(repr(key) for key in unknown)
            + "; accepted keys: "
            + ", ".join(sorted(DISTILLATION_KEYS))
        )

    enabled = bool(block.get("ENABLED", False)) if require_teachers is None else require_teachers
    teachers = block.get("TEACHERS") or []
    if enabled:
        if not isinstance(teachers, (list, tuple)) or not teachers:
            raise ImproperlyConfigured(
                "DISTILLATION.ENABLED is true but DISTILLATION.TEACHERS is empty; a "
                "distilled run requires at least one teacher"
            )
    if teachers and not isinstance(teachers, (list, tuple)):
        raise ImproperlyConfigured("DISTILLATION.TEACHERS must be a list of teacher specs")
    if len(teachers) > MAX_TEACHERS:
        raise ImproperlyConfigured(
            f"DISTILLATION.TEACHERS has {len(teachers)} entries; the per-step cost is "
            f"N peers x M teachers contrastive terms, and more than {MAX_TEACHERS} is a "
            "configuration mistake rather than a design"
        )

    out_dim = block.get("OUT_DIM", 64)
    if isinstance(out_dim, bool) or not isinstance(out_dim, int) or out_dim < 1:
        raise ImproperlyConfigured(
            f"DISTILLATION.OUT_DIM must be a positive integer, got {out_dim!r}"
        )

    tau = _as_float(block, "TAU", 0.07, owner)
    tau_min = _as_float(block, "TAU_MIN", 0.01, owner)
    tau_max = _as_float(block, "TAU_MAX", 1.0, owner)
    if not tau_min > 0.0:
        raise ImproperlyConfigured("DISTILLATION.TAU_MIN must be strictly positive")
    if not tau_max >= tau_min:
        raise ImproperlyConfigured("DISTILLATION.TAU_MAX must be >= TAU_MIN")
    if not tau > 0.0:
        raise ImproperlyConfigured("DISTILLATION.TAU must be strictly positive")

    if _as_float(block, "BETA", 1.0, owner) < 0.0:
        raise ImproperlyConfigured("DISTILLATION.BETA must be >= 0")

    mode = str(block.get("SIGMA_MODE", "quality_softmax"))
    if mode not in SIGMA_MODE_NAMES:
        raise ImproperlyConfigured(
            "DISTILLATION.SIGMA_MODE must be 'uniform', 'teacher_softmax', or "
            f"'quality_softmax', got {mode!r}"
        )

    sigma_floor = _as_float(block, "SIGMA_FLOOR", 0.0, owner)
    sigma_cap = _as_float(block, "SIGMA_CAP", 1.0, owner)
    if not 0.0 <= sigma_floor <= sigma_cap:
        raise ImproperlyConfigured("DISTILLATION.SIGMA_FLOOR must be within [0, SIGMA_CAP]")
    if not 0.0 < sigma_cap <= 1.0:
        raise ImproperlyConfigured("DISTILLATION.SIGMA_CAP must be in (0, 1]")
    count = len(teachers)
    if count and sigma_floor * count > 1.0:
        raise ImproperlyConfigured(
            f"DISTILLATION.SIGMA_FLOOR={sigma_floor:g} x {count} teachers exceeds 1.0; "
            "no normalized sigma vector exists"
        )
    if count and sigma_cap * count < 1.0:
        raise ImproperlyConfigured(
            f"DISTILLATION.SIGMA_CAP={sigma_cap:g} x {count} teachers is below 1.0; "
            "no normalized sigma vector exists"
        )

    decay = _as_float(block, "SIGMA_DECAY", 0.9, owner)
    if not 0.0 <= decay < 1.0:
        raise ImproperlyConfigured("DISTILLATION.SIGMA_DECAY must be in [0, 1)")

    for key in ("LAMBDA_COOP", "LAMBDA_IND", "LAMBDA_REG", "LAMBDA_CD"):
        value = _as_float(block, key, DISTILLATION_DEFAULTS[key], owner)
        if value < 0.0:
            raise ImproperlyConfigured(f"DISTILLATION.{key} must be >= 0, got {value!r}")
    if (
        _as_float(block, "LAMBDA_COOP", 0.0, owner) == 0.0
        and _as_float(block, "LAMBDA_IND", 1.0, owner) == 0.0
        and _as_float(block, "LAMBDA_CD", 1.0, owner) == 0.0
    ):
        raise ImproperlyConfigured(
            "DISTILLATION.LAMBDA_COOP, LAMBDA_IND, and LAMBDA_CD are all zero; the run "
            "would train nothing but the regularizer"
        )

    margin = _as_float(block, "PEER_MARGIN", 0.25, owner)
    if not 0.0 <= margin <= 1.0:
        raise ImproperlyConfigured("DISTILLATION.PEER_MARGIN must be in [0, 1]")

    for key in ("WARMUP_STEPS", "RAMP_STEPS"):
        value = block.get(key, 0)
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ImproperlyConfigured(f"DISTILLATION.{key} must be a non-negative integer")

    views = block.get("VIEWS", {})
    if not isinstance(views, Mapping):
        raise ImproperlyConfigured("DISTILLATION.VIEWS must be a mapping")
    for key, spec in views.items():
        if not isinstance(spec, Mapping):
            raise ImproperlyConfigured(
                f"DISTILLATION.VIEWS[{key!r}] must be a mapping, got {type(spec).__name__}"
            )
        kind = str(spec.get("kind", spec.get("type", "identity")))
        if kind not in VIEW_KIND_NAMES:
            raise ImproperlyConfigured(
                f"DISTILLATION.VIEWS[{key!r}].kind is {kind!r}; accepted: identity, "
                "linear, fixed_random, hidden, distribution"
            )

    regularizers = block.get("REGULARIZERS", [])
    if not isinstance(regularizers, (list, tuple)):
        raise ImproperlyConfigured("DISTILLATION.REGULARIZERS must be a list of specs")

    _validate_conformance(block, teachers)


def _validate_conformance(block: Mapping[str, Any], teachers: Any) -> None:
    conformance = block.get("CONFORMANCE", {})
    if not isinstance(conformance, Mapping):
        raise ImproperlyConfigured("DISTILLATION.CONFORMANCE must be a mapping")
    for key in sorted(set(conformance) - CONFORMANCE_KEYS):
        raise ImproperlyConfigured(
            f"DISTILLATION.CONFORMANCE.{key} is not a known key; accepted: "
            + ", ".join(sorted(CONFORMANCE_KEYS))
        )
    owner = "DISTILLATION.CONFORMANCE."
    mode = str(conformance.get("MODE", "hybrid"))
    if mode not in CONFORMANCE_MODES:
        raise ImproperlyConfigured(
            "DISTILLATION.CONFORMANCE.MODE must be 'declared', 'audit', or 'hybrid', "
            f"got {mode!r}"
        )

    trust_init = _as_float(conformance, "TRUST_INIT", 1.0, owner)
    trust_floor = _as_float(conformance, "TRUST_FLOOR", 0.0, owner)
    trust_min = _as_float(conformance, "TRUST_MIN", 0.05, owner)
    if not 0.0 <= trust_floor <= trust_init <= 1.0:
        raise ImproperlyConfigured(
            "DISTILLATION.CONFORMANCE requires 0 <= TRUST_FLOOR <= TRUST_INIT <= 1, "
            f"got TRUST_FLOOR={trust_floor:g}, TRUST_INIT={trust_init:g}"
        )
    if not 0.0 < trust_min < 1.0:
        raise ImproperlyConfigured(
            f"DISTILLATION.CONFORMANCE.TRUST_MIN must be in (0, 1), got {trust_min:g}"
        )
    if trust_min < trust_floor:
        raise ImproperlyConfigured(
            f"DISTILLATION.CONFORMANCE.TRUST_MIN={trust_min:g} is below "
            f"TRUST_FLOOR={trust_floor:g}; a teacher excluded at TRUST_MIN could never "
            "reach the floor that is supposed to bound it"
        )
    trust_decay = _as_float(conformance, "TRUST_DECAY", 0.5, owner)
    if not 0.0 < trust_decay <= 1.0:
        raise ImproperlyConfigured(
            f"DISTILLATION.CONFORMANCE.TRUST_DECAY must be in (0, 1], got {trust_decay:g}"
        )
    trust_restore = _as_float(conformance, "TRUST_RESTORE", 0.05, owner)
    if not 0.0 <= trust_restore <= 1.0:
        raise ImproperlyConfigured(
            f"DISTILLATION.CONFORMANCE.TRUST_RESTORE must be in [0, 1], got {trust_restore:g}"
        )

    artifacts = conformance.get("AUDIT_ARTIFACTS", {})
    if not isinstance(artifacts, Mapping):
        raise ImproperlyConfigured(
            "DISTILLATION.CONFORMANCE.AUDIT_ARTIFACTS must map teacher name -> path"
        )
    declared_names = {
        name for name in (_teacher_name(spec) for spec in (teachers or ())) if name is not None
    }
    for teacher_name, artifact_path in artifacts.items():
        if str(teacher_name) not in declared_names:
            raise ImproperlyConfigured(
                f"DISTILLATION.CONFORMANCE.AUDIT_ARTIFACTS names teacher "
                f"{teacher_name!r}, which is not in DISTILLATION.TEACHERS; an audit of a "
                "teacher that is not configured can never be applied"
            )
        if not Path(str(artifact_path)).is_file():
            raise ImproperlyConfigured(
                f"DISTILLATION.CONFORMANCE.AUDIT_ARTIFACTS[{teacher_name!r}] points at "
                f"{artifact_path!r}, which is not a file; a missing audit is not an "
                "empty audit"
            )

    if mode == "audit" and bool(conformance.get("ENABLED", True)):
        declared_audit = {
            name
            for spec in (teachers or ())
            for name in [_teacher_name(spec)]
            if name is not None
            and (
                spec.get("audit") if isinstance(spec, Mapping) else getattr(spec, "audit", None)
            )
        }
        uncovered = declared_names - declared_audit - {str(name) for name in artifacts}
        if uncovered:
            raise ImproperlyConfigured(
                "DISTILLATION.CONFORMANCE.MODE is 'audit' but teacher(s) "
                + ", ".join(sorted(uncovered))
                + " declare no auditor; set the spec's 'audit' field, add an "
                "AUDIT_ARTIFACTS entry, or switch MODE to 'hybrid'"
            )


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay onto base; nested mappings merge, everything else is replaced.

    Imported lazily from ``conf`` would create an import cycle (``conf`` installs
    this module's defaults at import time), so the one merge rule is restated here
    with identical semantics: a mapping merges key by key, any other value replaces.
    """
    from ..conf import _deep_merge as merge

    return merge(base, dict(overlay))


@dataclass(frozen=True)
class DistillationConfig:
    """The resolved ``DISTILLATION`` block for one run.

    Frozen and complete: every key in ``DISTILLATION_DEFAULTS`` is present with its
    effective value, so ``build()`` reads one object and one object only.
    ``from_spec`` is the only constructor a caller should use; the dataclass
    constructor is what it calls after merging.
    """

    enabled: bool = False
    teachers: tuple[Mapping[str, Any], ...] = ()
    out_dim: int = 64
    tau: float = 0.07
    tau_min: float = 0.01
    tau_max: float = 1.0
    symmetric: bool = True
    beta: float = 1.0
    sigma_mode: str = "quality_softmax"
    sigma_floor: float = 0.0
    sigma_cap: float = 1.0
    sigma_decay: float = 0.9
    lambda_coop: float = 0.0
    lambda_ind: float = 1.0
    lambda_reg: float = 0.0
    lambda_cd: float = 1.0
    peer_margin: float = 0.25
    warmup_steps: int = 0
    ramp_steps: int = 0
    regularizers: tuple[Any, ...] = ()
    views: Mapping[str, Any] = field(default_factory=dict)
    strict: bool = True
    conformance: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_spec(
        cls,
        spec: "bool | Mapping[str, Any] | DistillationConfig | None",
        *,
        settings: Any,
    ) -> "DistillationConfig":
        """Resolve the argument, then the settings block, then the defaults.

        ``ENABLED`` sits outside the merge precedence: it is resolved first from the
        resolution table, then the merge fills in everything else. It is popped
        from the explicit mapping *before* merging, so a settings-block ``ENABLED``
        can never leak in underneath an explicit one and ``True`` / ``False`` stay
        unconditional.

        Raises ``ValidationError`` on an unknown key (naming it and listing the
        accepted set) and ``ImproperlyConfigured`` on an invalid value.
        """
        if isinstance(spec, DistillationConfig):
            return spec
        raw_block = settings.get("DISTILLATION", {}) if settings is not None else {}
        settings_block = copy.deepcopy(dict(raw_block or {}))

        if spec is False:
            enabled, explicit = False, {}
        elif spec is True:
            enabled, explicit = True, {}
        elif spec is None:
            enabled, explicit = bool(settings_block.get("ENABLED", False)), {}
        elif isinstance(spec, Mapping):
            explicit = copy.deepcopy({str(key): value for key, value in spec.items()})
            # A mapping without ENABLED means on; one with ENABLED honours it.
            enabled = bool(explicit.pop("ENABLED", True))
        else:
            raise ValidationError(
                "distillation must be a bool, a mapping, a DistillationConfig, or None; "
                f"got {type(spec).__name__}"
            )

        merged = _deep_merge(
            _deep_merge(copy.deepcopy(DISTILLATION_DEFAULTS), settings_block), explicit
        )
        # The resolved toggle wins, never the merged one.
        merged.pop("ENABLED", None)
        validate_block({**merged, "ENABLED": enabled}, require_teachers=enabled)

        values = {key.lower(): value for key, value in merged.items()}
        values["teachers"] = tuple(copy.deepcopy(item) for item in (values.get("teachers") or ()))
        values["regularizers"] = tuple(
            copy.deepcopy(item) for item in (values.get("regularizers") or ())
        )
        values["views"] = copy.deepcopy(dict(values.get("views") or {}))
        values["conformance"] = copy.deepcopy(dict(values.get("conformance") or {}))
        return cls(enabled=enabled, **values)

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe form with upper-case keys, ``from_spec``-compatible.

        Teachers appear as their spec mappings: weights and audit artifacts are
        recorded by path, never by content, because the checkpoint already stores
        teacher digests.
        """
        payload: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in ("teachers", "regularizers"):
                value = [copy.deepcopy(entry) for entry in value]
            elif isinstance(value, Mapping):
                value = copy.deepcopy(dict(value))
            payload[item.name.upper()] = value
        return payload

    def objective_kwargs(self) -> dict[str, Any]:
        """The keyword arguments ``ObjectiveConfig`` takes."""
        return {
            "tau": self.tau,
            "tau_min": self.tau_min,
            "tau_max": self.tau_max,
            "symmetric": self.symmetric,
            "beta": self.beta,
            "sigma_mode": self.sigma_mode,
            "lambda_coop": self.lambda_coop,
            "lambda_ind": self.lambda_ind,
            "lambda_reg": self.lambda_reg,
            "lambda_cd": self.lambda_cd,
            "peer_margin": self.peer_margin,
            "warmup_steps": self.warmup_steps,
            "ramp_steps": self.ramp_steps,
        }
