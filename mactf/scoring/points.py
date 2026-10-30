"""The 0-100 points scale (document 11 sections 2-6).

Points are a second projection of the same ledger that feeds the docx score. They
do not replace ``final_score``: ``final_score`` is an unbounded training signal
comparable across resumes, ``final_points`` is a bounded rubric score comparable
across episodes and peers, and the two are allowed to disagree. An episode can
have ``final_score = +12.0`` and ``final_points = 0.0`` (void) at the same time.

This module imports only ``mactf/rules`` and ``mactf/scoring``. The guards that
decide whether a rule was *earned* live in :mod:`mactf.scoring.guards`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..exceptions import ImproperlyConfigured
from ..rules.catalog import CATALOG, NON_OFFSETTABLE, PENALTIES, REWARDS
from ..rules.severity import Severity
from .normalize import MAX_POINTS, MIN_POINTS

__all__ = [
    "DEFAULT_PENALTY_WEIGHTS",
    "DEFAULT_REWARD_WEIGHTS",
    "DEFAULT_SEVERITY_FRACTIONS",
    "DEFAULT_POINTS_TABLE",
    "DenialRecord",
    "MAX_POINTS",
    "MIN_POINTS",
    "PointsBreakdown",
    "PointsTable",
    "RulePoints",
    "SeverityPoints",
    "TABLE_VERSION",
    "finalize_points",
    "penalty_instance_points",
    "reward_instance_points",
]

TABLE_VERSION = "1.0.0"

#: Rewards: a partition of the 0-100 scale (section 3.1). The sum is exactly 100.
DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "R1": 25.0, "R2": 12.0, "R3": 2.0, "R4": 3.0, "R5": 3.0,
    "R6": 5.0, "R7": 5.0, "R8": 6.0, "R9": 3.0, "R10": 8.0,
    "R11": 5.0, "R12": 4.0, "R13": 6.0, "R14": 5.0, "R15": 3.0,
    "R16": 4.0, "R17": 1.0,
}

#: Penalties: sized against the reward scale (section 3.2).
DEFAULT_PENALTY_WEIGHTS: dict[str, float] = {
    "P1": 40.0, "P2": 40.0, "P3": 20.0, "P4": 5.0, "P5": 25.0,
    "P6": 6.0, "P7": 6.0, "P8": 5.0, "P9": 15.0, "P10": 15.0,
    "P11": 4.0, "P12": 4.0, "P13": 60.0, "P14": 60.0, "P15": 10.0,
    "P16": 25.0, "P17": 100.0, "P18": 10.0, "P19": 10.0, "P20": 100.0,
    "P21": 100.0, "P22": 100.0, "P23": 12.0,
}

DEFAULT_SEVERITY_FRACTIONS: dict[str, float] = {
    "S0": 0.00,
    "S1": 0.05,
    "S2": 0.15,
    "S3": 0.40,
    "S4": 1.00,
}

#: Penalties that are warning-first (framework.md W1), all of which must stay below 10.
_WARNING_FIRST_PENALTIES = ("P4", "P6", "P7", "P8", "P11", "P12")


def _fmt(value: float) -> str:
    return f"{value:g}"


@dataclass(frozen=True)
class SeverityPoints:
    """Maps a severity tier to the fraction of a rule's weight it carries.

    framework.md 4.3.3 fixes the ordering and the cap, not the numbers. These are
    the operator's numbers, recorded so they are auditable rather than scattered
    through scoring code. The fractions are deliberately superlinear because the
    framework distinguishes a typo (S1) from a fabricated citation (S3) from a
    fabricated safety result (S4) as qualitatively different failures.
    """

    fractions: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SEVERITY_FRACTIONS)
    )

    def fraction(self, severity: Severity | str | int) -> float:
        """Look up the fraction, raising rather than defaulting to 1.0."""
        label = Severity.parse(severity).label
        try:
            return float(self.fractions[label])
        except KeyError as exc:
            raise ImproperlyConfigured(f"SEVERITY_POINTS has no entry for {label}") from exc

    def validate(self) -> list[str]:
        """Ordering and bounds, as framework.md 4.3.3 requires."""
        problems: list[str] = []
        missing = [f"S{i}" for i in range(5) if f"S{i}" not in self.fractions]
        if missing:
            return [f"SCORING.POINTS.SEVERITY_FRACTIONS is missing {', '.join(missing)}"]
        ordered = [float(self.fractions[f"S{i}"]) for i in range(5)]
        if ordered != sorted(ordered):
            problems.append("severity point fractions must be non-decreasing S0..S4")
        if ordered[0] != 0.0:
            problems.append("S0 must carry zero points; it is informational only")
        if ordered[-1] != 1.0:
            problems.append("S4 must carry the full rule weight")
        if any(not 0.0 <= value <= 1.0 for value in ordered):
            problems.append("severity point fractions must be within [0, 1]")
        return problems

    def as_dict(self) -> dict[str, float]:
        return {label: float(self.fractions[label]) for label in sorted(self.fractions)}


@dataclass(frozen=True)
class PointsTable:
    """The complete rubric: one weight per rule ID, plus the severity mapping."""

    rewards: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_REWARD_WEIGHTS))
    penalties: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_PENALTY_WEIGHTS))
    severity: SeverityPoints = field(default_factory=SeverityPoints)
    warning_notional: float = 2.0
    table_version: str = TABLE_VERSION

    # -- derived ---------------------------------------------------------
    @property
    def total_reward_points(self) -> float:
        return float(sum(self.rewards.values()))

    @property
    def reward_total(self) -> float:
        return self.total_reward_points

    def weight(self, rule_id: str) -> float:
        """The rule's full weight. Raises for a rule with no entry."""
        if rule_id in self.rewards:
            return float(self.rewards[rule_id])
        if rule_id in self.penalties:
            return float(self.penalties[rule_id])
        raise ImproperlyConfigured(f"the points table has no weight for rule {rule_id!r}")

    def max_reward(self) -> float:
        return max((float(value) for value in self.rewards.values()), default=0.0)

    # -- validation ------------------------------------------------------
    def validate(self) -> list[str]:
        """Everything this table promises, checked in one place.

        Messages name the settings key and the offending value, in the order
        document 11 section 9.2 lists the checks.
        """
        problems: list[str] = []
        reward_ids = [rule.id for rule in REWARDS]
        penalty_ids = [rule.id for rule in PENALTIES]

        for rule_id in reward_ids:
            if rule_id not in self.rewards:
                problems.append(f"SCORING.POINTS.REWARD_WEIGHTS is missing {rule_id}")
        for rule_id in penalty_ids:
            if rule_id not in self.penalties:
                problems.append(f"SCORING.POINTS.PENALTY_WEIGHTS is missing {rule_id}")
        for rule_id in self.rewards:
            if rule_id not in reward_ids:
                problems.append(f"SCORING.POINTS.REWARD_WEIGHTS has unknown rule {rule_id}")
        for rule_id in self.penalties:
            if rule_id not in penalty_ids:
                problems.append(f"SCORING.POINTS.PENALTY_WEIGHTS has unknown rule {rule_id}")
        if problems:
            # Everything below indexes by rule id; stop at the structural errors.
            return problems

        for rule_id, value in self.rewards.items():
            if not _finite(value) or float(value) <= 0.0:
                problems.append(
                    f"SCORING.POINTS.REWARD_WEIGHTS: {rule_id} ({value!r}) must be > 0"
                )
        for rule_id, value in self.penalties.items():
            if not _finite(value) or not 0.0 < float(value) <= MAX_POINTS:
                problems.append(
                    f"SCORING.POINTS.PENALTY_WEIGHTS: {rule_id} ({value!r}) must be "
                    f"within (0, {_fmt(MAX_POINTS)}]"
                )
        if problems:
            return problems

        total = self.total_reward_points
        if abs(total - MAX_POINTS) > 1e-9:
            problems.append(
                f"SCORING.POINTS.REWARD_WEIGHTS must sum to {MAX_POINTS:.1f}, got {total:g}"
            )

        largest = self.max_reward()
        for rule_id in sorted(NON_OFFSETTABLE, key=lambda item: int(item[1:])):
            weight = float(self.penalties[rule_id])
            if weight < largest:
                problems.append(
                    f"SCORING.POINTS.PENALTY_WEIGHTS: {rule_id} ({_fmt(weight)}) must be "
                    f">= the largest reward ({_fmt(largest)})"
                )

        problems.extend(self.severity.validate())

        r = {key: float(value) for key, value in self.rewards.items()}
        p = {key: float(value) for key, value in self.penalties.items()}
        if not r["R10"] > r["R4"] + r["R5"]:
            problems.append("SCORING.POINTS.REWARD_WEIGHTS: R10 must exceed R4 + R5")
        if r["R17"] > 2.0:
            problems.append("SCORING.POINTS.REWARD_WEIGHTS: R17 must be <= 2")
        if r["R1"] < max(r.values()):
            problems.append("SCORING.POINTS.REWARD_WEIGHTS: R1 must be the largest reward")
        if r["R9"] > r["R2"]:
            problems.append(
                "SCORING.POINTS.REWARD_WEIGHTS: R9 (T6) must not exceed R2 (T1)"
            )
        if p["P1"] != p["P2"]:
            problems.append("SCORING.POINTS.PENALTY_WEIGHTS: P1 and P2 must be equal")
        for rule_id in ("P13", "P14"):
            if p[rule_id] < 50.0:
                problems.append(
                    f"SCORING.POINTS.PENALTY_WEIGHTS: {rule_id} ({_fmt(p[rule_id])}) must be >= 50"
                )
        for rule_id in ("P17", "P20", "P21"):
            if p[rule_id] != MAX_POINTS:
                problems.append(
                    f"SCORING.POINTS.PENALTY_WEIGHTS: {rule_id} ({_fmt(p[rule_id])}) must "
                    f"saturate the scale at {_fmt(MAX_POINTS)}"
                )
        for rule_id in _WARNING_FIRST_PENALTIES:
            if p[rule_id] >= 10.0:
                problems.append(
                    f"SCORING.POINTS.PENALTY_WEIGHTS: {rule_id} ({_fmt(p[rule_id])}) is "
                    "warning-first and must be below 10"
                )
        if not _finite(self.warning_notional) or float(self.warning_notional) < 0.0:
            problems.append(
                f"SCORING.POINTS.WARNING_NOTIONAL must be >= 0, got {self.warning_notional!r}"
            )
        return problems

    def assert_valid(self) -> "PointsTable":
        problems = self.validate()
        if problems:
            raise ImproperlyConfigured("; ".join(problems))
        return self

    # -- construction ----------------------------------------------------
    @classmethod
    def from_settings(
        cls, settings: Mapping[str, Any] | None = None, *, validate: bool = True
    ) -> "PointsTable":
        """Read ``SCORING.POINTS``. Accepts the whole settings, ``SCORING``, or ``POINTS``.

        Absent keys take the documented defaults; keys that are present are used
        as given, so a partial ``REWARD_WEIGHTS`` is reported as missing rules
        instead of being silently topped up.
        """
        block: Mapping[str, Any] = settings or {}
        wrapped = False
        if "SCORING" in block and isinstance(block["SCORING"], Mapping):
            block = block["SCORING"]
            wrapped = True
        if "POINTS" in block and isinstance(block["POINTS"], Mapping):
            block = block["POINTS"]
        elif wrapped or "PRECEDENCE_MODE" in block:
            # SCORING.REWARD_WEIGHTS is the docx r_* table, not the points table.
            block = {}
        table = cls(
            rewards={
                str(k): float(v)
                for k, v in (block.get("REWARD_WEIGHTS") or DEFAULT_REWARD_WEIGHTS).items()
            },
            penalties={
                str(k): float(v)
                for k, v in (block.get("PENALTY_WEIGHTS") or DEFAULT_PENALTY_WEIGHTS).items()
            },
            severity=SeverityPoints(
                {
                    str(k): float(v)
                    for k, v in (
                        block.get("SEVERITY_FRACTIONS") or DEFAULT_SEVERITY_FRACTIONS
                    ).items()
                }
            ),
            warning_notional=float(block.get("WARNING_NOTIONAL", 2.0)),
            table_version=str(block.get("TABLE_VERSION", TABLE_VERSION)),
        )
        return table.assert_valid() if validate else table

    def as_dict(self) -> dict[str, Any]:
        return {
            "table_version": self.table_version,
            "rewards": {k: float(v) for k, v in self.rewards.items()},
            "penalties": {k: float(v) for k, v in self.penalties.items()},
            "severity_fractions": self.severity.as_dict(),
            "warning_notional": float(self.warning_notional),
            "reward_total": self.total_reward_points,
            "max_points": MAX_POINTS,
            "min_points": MIN_POINTS,
        }

    to_dict = as_dict

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PointsTable":
        return cls(
            rewards={str(k): float(v) for k, v in payload["rewards"].items()},
            penalties={str(k): float(v) for k, v in payload["penalties"].items()},
            severity=SeverityPoints(
                {str(k): float(v) for k, v in payload["severity_fractions"].items()}
            ),
            warning_notional=float(payload.get("warning_notional", 2.0)),
            table_version=str(payload.get("table_version", TABLE_VERSION)),
        )

    def describe(self) -> str:
        return (
            f"points table {self.table_version}: {len(self.rewards)} rewards "
            f"(sum {self.total_reward_points:g}), {len(self.penalties)} penalties"
        )


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


DEFAULT_POINTS_TABLE = PointsTable()


# ---------------------------------------------------------------------------
# Accumulation (section 3.3)
# ---------------------------------------------------------------------------
def reward_instance_points(weight: float, fraction: float, instance: int) -> float:
    """Diminishing returns for repeats: framework.md 4.3.4.

    ``instance`` is 1-based. A second R4 is worth half the first, a third a
    quarter, and the rule can never exceed its own weight.
    """
    if instance < 1:
        raise ValueError(f"instance is 1-based, got {instance}")
    return weight * fraction * (0.5 ** (instance - 1))


def penalty_instance_points(weight: float, fraction: float, instance: int) -> float:
    """Linear accumulation for penalties: framework.md 4.3.4.

    Repeated instances escalate; the linear form matches ``continuous.py``'s
    per-rule summation, so a repeat costs what the ledger says it costs. The
    rule's total is capped at ``weight`` by the engine, not here.
    """
    if instance < 1:
        raise ValueError(f"instance is 1-based, got {instance}")
    return weight * fraction


# ---------------------------------------------------------------------------
# Breakdown (section 4.7)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DenialRecord:
    """Why a reward earned nothing, or a penalty was suppressed by its guard."""

    rule_id: str
    reason: str
    missing: tuple[str, ...] = ()
    clause: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "reason": self.reason,
            "missing": list(self.missing),
            "clause": self.clause,
        }

    to_dict = as_dict

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DenialRecord":
        return cls(
            rule_id=str(payload["rule_id"]),
            reason=str(payload.get("reason", "")),
            missing=tuple(payload.get("missing", ())),
            clause=str(payload.get("clause", "")),
        )


@dataclass(frozen=True)
class RulePoints:
    """What one rule contributed."""

    rule_id: str
    name: str
    weight: float
    fraction: float
    instances: int
    points: float
    capped: bool = False
    clause: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "weight": self.weight,
            "fraction": self.fraction,
            "instances": self.instances,
            "points": self.points,
            "capped": self.capped,
            "clause": self.clause,
        }

    to_dict = as_dict

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RulePoints":
        return cls(
            rule_id=str(payload["rule_id"]),
            name=str(payload.get("name", "")),
            weight=float(payload["weight"]),
            fraction=float(payload["fraction"]),
            instances=int(payload["instances"]),
            points=float(payload["points"]),
            capped=bool(payload.get("capped", False)),
            clause=str(payload.get("clause", "")),
        )


@dataclass(frozen=True)
class PointsBreakdown:
    """The full derivation of one peer episode's points."""

    reward_points: float
    penalty_points: float
    warning_points: float
    gross: float
    final_points: float
    void: bool
    void_rules: tuple[str, ...] = ()
    rewards: tuple[RulePoints, ...] = ()
    penalties: tuple[RulePoints, ...] = ()
    warnings: tuple[RulePoints, ...] = ()
    denied: tuple[DenialRecord, ...] = ()
    suppressed_by_guard: tuple[DenialRecord, ...] = ()
    suppressed_by_precedence: tuple[str, ...] = ()
    severity_fractions: Mapping[str, float] = field(default_factory=dict)
    table_version: str = TABLE_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "reward_points": self.reward_points,
            "penalty_points": self.penalty_points,
            "warning_points": self.warning_points,
            "gross": self.gross,
            "final_points": self.final_points,
            "void": self.void,
            "void_rules": list(self.void_rules),
            "rewards": [item.as_dict() for item in self.rewards],
            "penalties": [item.as_dict() for item in self.penalties],
            "warnings": [item.as_dict() for item in self.warnings],
            "denied": [item.as_dict() for item in self.denied],
            "suppressed_by_guard": [item.as_dict() for item in self.suppressed_by_guard],
            "suppressed_by_precedence": list(self.suppressed_by_precedence),
            "severity_fractions": dict(self.severity_fractions),
            "table_version": self.table_version,
        }

    to_dict = as_dict

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PointsBreakdown":
        return cls(
            reward_points=float(payload["reward_points"]),
            penalty_points=float(payload["penalty_points"]),
            warning_points=float(payload.get("warning_points", 0.0)),
            gross=float(payload["gross"]),
            final_points=float(payload["final_points"]),
            void=bool(payload["void"]),
            void_rules=tuple(payload.get("void_rules", ())),
            rewards=tuple(RulePoints.from_dict(i) for i in payload.get("rewards", ())),
            penalties=tuple(RulePoints.from_dict(i) for i in payload.get("penalties", ())),
            warnings=tuple(RulePoints.from_dict(i) for i in payload.get("warnings", ())),
            denied=tuple(DenialRecord.from_dict(i) for i in payload.get("denied", ())),
            suppressed_by_guard=tuple(
                DenialRecord.from_dict(i) for i in payload.get("suppressed_by_guard", ())
            ),
            suppressed_by_precedence=tuple(payload.get("suppressed_by_precedence", ())),
            severity_fractions={
                str(k): float(v) for k, v in payload.get("severity_fractions", {}).items()
            },
            table_version=str(payload.get("table_version", TABLE_VERSION)),
        )


def finalize_points(breakdown: PointsBreakdown | float, *, void: bool) -> float:
    """The only place ``final_points`` is computed.

    ``void`` is read, never decided here: it is ``bool(violated_non_offsettable)``
    from the pipeline, unioned with ``branch_void`` in a hierarchical run. A void
    episode is pinned to ``MIN_POINTS`` -- the worst result on a 0-100 scale is
    zero -- while ``final_score`` keeps carrying the negative value and the floor.
    """
    if void:
        return MIN_POINTS
    gross = breakdown if isinstance(breakdown, (int, float)) else breakdown.gross
    return min(MAX_POINTS, max(MIN_POINTS, float(gross)))


def rule_name(rule_id: str) -> str:
    return CATALOG[rule_id].name if rule_id in CATALOG else rule_id
