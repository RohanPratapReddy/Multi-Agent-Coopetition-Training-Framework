"""WeightGuard: the compliance boundary of a weight strategy (document 17 section 10.4)."""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..exceptions import ComplianceError, ImproperlyConfigured, WeightGuardViolation
from .aggregation import Aggregator, _project_bounded
from .report import AggregationReport
from .weights import (
    BUILTIN_WEIGHT_TYPES,
    KappaSoftmaxAverage,
    WeightTraits,
    traits_of,
)

log = logging.getLogger("mactf.weights")

GUARD_MODES = ("repair", "fallback", "enforce")
GUARD_SCOPES = ("custom", "all", "off")
_GUARD_KEYS = frozenset({"MODE", "SCOPE", "TOLERANCE", "PROBE"})

CODES: dict[str, str] = {
    "U001": "a weight is not a finite number",
    "U002": "a weight is negative",
    "U003": "the weights do not name exactly the peers of the group",
    "U004": "an ineligible or trimmed peer carries weight",
    "U005": "a participant is outside [floor, cap], or the weights do not sum to 1",
    "U006": "the strategy reported fallback without declaring may_fallback",
    "U007": "apply() raised",
}
_HARD = frozenset({"U001", "U002", "U003", "U006", "U007"})


def _exact(value: Any) -> Any:
    """``value`` as an exact ``bool``/``int``/``float`` when it is one (or a subclass of one).

    A ``float`` subclass can override its comparisons and ``__float__``: the guard would
    judge one number and the aggregator would apply another. Anything else is returned as
    it is, for ``inspect`` to refuse.
    """
    if isinstance(value, bool):
        return bool(int.__int__(value))
    if isinstance(value, int):
        return int.__int__(value)
    if isinstance(value, float):
        return float.__float__(value)
    return value


def read_once(report: Any) -> AggregationReport:
    """An exact :class:`AggregationReport` holding one reading of every field of ``report``.

    The weights the guard inspects are then the weights the trainer writes, whatever
    properties, mapping subclasses or number subclasses the strategy's report is made of.
    """
    values: dict[str, Any] = {}
    for item in dataclasses.fields(AggregationReport):
        value = getattr(report, item.name)
        if item.name in ("weights", "raw_weights"):
            value = {str(key): _exact(number) for key, number in dict(value).items()}
        elif item.name in ("adjustments",):
            value = [str(entry) for entry in list(value)]
        elif item.name in ("gated_peers", "trimmed_peers", "order"):
            value = tuple(str(entry) for entry in list(value))
        elif item.name == "fallback":
            value = bool(value)
        elif item.name == "signal_source":
            value = str(value)
        elif item.name in ("kappa", "entropy"):
            value = _exact(value)
        values[item.name] = value
    return AggregationReport(**values)


@dataclass(frozen=True)
class Violation:
    code: str
    peer: str | None
    detail: str

    @property
    def hard(self) -> bool:
        return self.code in _HARD

    def __str__(self) -> str:
        where = "" if self.peer is None else f" [{self.peer}]"
        return f"{self.code}{where}: {self.detail}"


@dataclass
class GuardOutcome:
    #: The report to use; ``None`` when ``action == "fallback"`` (the caller runs the fallback).
    report: AggregationReport | None
    violations: tuple[Violation, ...]
    action: str                       # "pass" | "repaired" | "fallback"


class WeightGuard:
    """Checks, and under ``repair`` fixes, what a weight strategy returned."""

    def __init__(
        self,
        *,
        mode: str = "repair",
        scope: str = "custom",
        tolerance: float = 1e-6,
        probe: bool = True,
    ) -> None:
        if mode not in GUARD_MODES:
            raise ImproperlyConfigured(
                f"E060: TRAINER.WEIGHT_GUARD.MODE must be one of {list(GUARD_MODES)}, got {mode!r}"
            )
        if scope not in GUARD_SCOPES:
            raise ImproperlyConfigured(
                f"E060: TRAINER.WEIGHT_GUARD.SCOPE must be one of {list(GUARD_SCOPES)}, got {scope!r}"
            )
        if not (isinstance(tolerance, (int, float)) and math.isfinite(tolerance) and 0.0 < tolerance < 0.1):
            raise ImproperlyConfigured(
                f"E060: TRAINER.WEIGHT_GUARD.TOLERANCE must be in (0, 0.1), got {tolerance!r}"
            )
        self.mode = mode
        self.scope = scope
        self.tolerance = float(tolerance)
        self.probe = bool(probe)

    # -- construction ----------------------------------------------------
    @classmethod
    def from_settings(cls, block: Mapping[str, Any] | None) -> "WeightGuard":
        raw = {str(key).upper(): value for key, value in dict(block or {}).items()}
        unknown = sorted(set(raw) - _GUARD_KEYS)
        if unknown:
            raise ImproperlyConfigured(
                f"E060: TRAINER.WEIGHT_GUARD has unknown keys {unknown}; accepted: {sorted(_GUARD_KEYS)}"
            )
        return cls(
            mode=str(raw.get("MODE", "repair")).lower(),
            scope=str(raw.get("SCOPE", "custom")).lower(),
            tolerance=raw.get("TOLERANCE", 1e-6),
            probe=bool(raw.get("PROBE", True)),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "scope": self.scope,
            "tolerance": self.tolerance,
            "probe": self.probe,
        }

    # -- scope -----------------------------------------------------------
    def covers(self, strategy: Any, aggregator: Aggregator | None = None) -> bool:
        """Whether ``strategy``'s output is checked.

        Under ``custom`` that is everything the framework did not write, **and** the default
        strategy when the aggregator it delegates to is a subclass: a custom
        ``Aggregator.compute_weights`` is custom code on the default path.
        """
        if self.scope == "off" or not traits_of(strategy).aggregates:
            return False
        if self.scope == "all":
            return True
        if not any(type(strategy) is built for built in BUILTIN_WEIGHT_TYPES):
            return True
        if isinstance(strategy, KappaSoftmaxAverage):
            bound = strategy.aggregator if strategy.aggregator is not None else aggregator
            return bound is not None and type(bound) is not Aggregator
        return False

    # -- the check -------------------------------------------------------
    @staticmethod
    def participants(
        report: AggregationReport, names: Sequence[str], eligible: Mapping[str, bool]
    ) -> list[str]:
        trimmed = set(report.trimmed_peers)
        return [
            name for name in sorted(names)
            if bool(eligible.get(name, True)) and name not in trimmed
        ]

    def inspect(
        self,
        report: AggregationReport,
        *,
        names: Sequence[str],
        eligible: Mapping[str, bool],
        floor: float,
        cap: float,
        traits: WeightTraits,
    ) -> list[Violation]:
        """Every violation in ``report``. Pure: nothing is changed."""
        tol = self.tolerance
        found: list[Violation] = []
        expected, got = set(names), set(report.weights)
        if got != expected:
            found.append(Violation(
                "U003", None,
                f"missing {sorted(expected - got)}, unexpected {sorted(got - expected)}",
            ))
            return found                         # nothing else is meaningful
        for name in sorted(names):
            value = report.weights[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                found.append(Violation("U001", name, f"weight is {value!r}"))
        if found:
            return found
        for name in sorted(names):
            if report.weights[name] < -tol:
                found.append(Violation("U002", name, f"weight is {report.weights[name]:.6g}"))
        if found:
            return found
        if report.fallback and not traits.may_fallback:
            found.append(Violation("U006", None, "report.fallback is True"))
            return found
        if report.fallback:
            return found                         # an intended reset: weights are informational
        participants = self.participants(report, names, eligible)
        chosen = set(participants)
        for name in sorted(names):
            if name not in chosen and report.weights[name] > tol:
                found.append(Violation(
                    "U004", name,
                    f"carries weight {report.weights[name]:.6g} but is ineligible or trimmed",
                ))
        if participants:
            for name in participants:
                value = report.weights[name]
                if value < floor - tol or value > cap + tol:
                    found.append(Violation(
                        "U005", name,
                        f"weight {value:.6g} is outside [{floor:g}, {cap:g}]",
                    ))
            total = sum(report.weights.values())
            if abs(total - 1.0) > max(tol, 1e-9) * max(1, len(names)):
                found.append(Violation("U005", None, f"weights sum to {total:.9g}, not 1"))
        return found

    def repair(
        self,
        report: AggregationReport,
        violations: Sequence[Violation],
        *,
        names: Sequence[str],
        eligible: Mapping[str, bool],
        floor: float,
        cap: float,
    ) -> AggregationReport:
        """The one lawful repair of soft violations: project onto the bounded simplex.

        Order: zero everyone who is not a participant; renormalize the participants'
        proposed weights (uniform when they sum to 0); project with ``_project_bounded``.
        Raises ``ComplianceError`` when the bounds are infeasible for this participant count;
        the caller then falls back.
        """
        participants = self.participants(report, names, eligible)
        if not participants:
            weights = {name: 0.0 for name in names}
        else:
            proposed = {name: max(0.0, float(report.weights[name])) for name in participants}
            total = sum(proposed.values())
            if total <= 0.0:
                proposed = {name: 1.0 / len(participants) for name in participants}
            else:
                proposed = {name: value / total for name, value in proposed.items()}
            projected = _project_bounded(proposed, floor=floor, cap=cap)
            weights = {name: float(projected.get(name, 0.0)) for name in names}
        codes = ",".join(sorted({item.code for item in violations}))
        gated = set(report.gated_peers) | {
            name for name in names if not bool(eligible.get(name, True))
        }
        return dataclasses.replace(
            report,
            weights=weights,
            adjustments=[*report.adjustments, f"weight_guard repaired weights ({codes})"],
            gated_peers=tuple(sorted(gated)),
        )

    def check(
        self,
        strategy: Any,
        report: AggregationReport,
        *,
        names: Sequence[str],
        eligible: Mapping[str, bool],
        floor: float,
        cap: float,
    ) -> GuardOutcome:
        label = getattr(strategy, "name", None) or type(strategy).__name__
        try:
            report = read_once(report)
        except Exception as error:  # noqa: BLE001 - an unreadable report is refused
            found = [Violation("U001", None, f"the report cannot be read: {type(error).__name__}: {error}")]
            if self.mode == "enforce":
                raise WeightGuardViolation(label, found) from error
            log.warning("W017: weight_update %r rejected (%s); falling back to kappa_softmax",
                        label, found[0])
            return GuardOutcome(None, tuple(found), "fallback")
        found = self.inspect(
            report, names=names, eligible=eligible, floor=floor, cap=cap,
            traits=traits_of(strategy),
        )
        if not found:
            return GuardOutcome(report, (), "pass")
        if self.mode == "enforce":
            raise WeightGuardViolation(label, found)
        if self.mode == "fallback" or any(item.hard for item in found):
            log.warning("W017: weight_update %r rejected (%s); falling back to kappa_softmax",
                        label, "; ".join(str(item) for item in found))
            return GuardOutcome(None, tuple(found), "fallback")
        try:
            repaired = self.repair(
                report, found, names=names, eligible=eligible, floor=floor, cap=cap
            )
        except ComplianceError as error:
            found.append(Violation("U005", None, f"repair impossible: {error}"))
            log.warning("W017: weight_update %r could not be repaired (%s); falling back", label, error)
            return GuardOutcome(None, tuple(found), "fallback")
        log.warning("W017: weight_update %r repaired (%s)", label,
                    "; ".join(str(item) for item in found))
        return GuardOutcome(repaired, tuple(found), "repaired")

    def run(
        self,
        strategy: Any,
        call: Callable[[Any], AggregationReport],
        *,
        names: Sequence[str],
        eligible: Mapping[str, bool],
        floor: float,
        cap: float,
    ) -> GuardOutcome:
        """Call the strategy and check what it returns. Covers ``apply`` raising (W-G11)."""
        label = getattr(strategy, "name", None) or type(strategy).__name__
        try:
            report = call(strategy)
        except Exception as error:  # noqa: BLE001 - third-party code; reported, never fatal
            found = Violation("U007", None, f"{type(error).__name__}: {error}")
            if self.mode == "enforce":
                raise WeightGuardViolation(label, [found]) from error
            log.warning("W017: weight_update %r raised (%s); falling back to kappa_softmax", label, found)
            return GuardOutcome(None, (found,), "fallback")
        return self.check(strategy, report, names=names, eligible=eligible, floor=floor, cap=cap)
