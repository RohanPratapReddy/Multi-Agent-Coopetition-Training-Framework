"""RollupGuard (document 17 section 8.6)."""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Mapping, Sequence

from ...exceptions import ComplianceGuardViolation, ImproperlyConfigured
from ...rl.functions.rollup import (
    ExplicitRollup,
    KappaWeightedRollup,
    MeanRollup,
    MinimumRollup,
    SumRollup,
    WorstCaseRollup,
)
from ...scoring.base import ScorerOutput
from .compliance import GUARD_MODES, GuardReport, GuardViolation

log = logging.getLogger("mactf.components")

#: The built-in strategies that union their children's voids by construction
#: (``rl/functions/rollup.py:111-143`` ``_output`` and ``_void_union``).
UNION_BY_CONSTRUCTION: tuple[type, ...] = (
    KappaWeightedRollup, MeanRollup, MinimumRollup, SumRollup, WorstCaseRollup,
)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (bool, str, bytes)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class RollupGuard:
    """Wraps a rollup and checks every output it produces.

    The wrapper answers to the same name as the strategy it wraps, so
    ``HierarchicalScorer.describe`` and the scorecard's ``rollup`` metadata read as before.
    """

    spec_family = "rollup"

    def __init__(self, inner: Any, *, mode: str = "enforce", label: str | None = None) -> None:
        if mode not in GUARD_MODES:
            raise ImproperlyConfigured(
                f"COMPLIANCE.GUARD_MODE must be one of {list(GUARD_MODES)}, got {mode!r}"
            )
        if not callable(getattr(inner, "rollup", None)):
            raise ImproperlyConfigured(
                f"RollupGuard needs an object with rollup(), got {type(inner).__name__}"
            )
        self.inner = inner
        self.mode = mode
        self.name = str(getattr(inner, "name", None) or type(inner).__name__)
        self.label = label or self.name
        self._reference = KappaWeightedRollup()
        #: The last 64 reports with a violation, newest last; empty on a clean run.
        self.reports: deque[GuardReport] = deque(maxlen=64)

    # -- the protocol method ------------------------------------------------------
    def rollup(
        self,
        parent: str,
        children: Sequence[ScorerOutput],
        *,
        weights: Mapping[str, float] | None = None,
        scores: Mapping[str, float] | None = None,
    ) -> ScorerOutput:
        failure: Exception | None = None
        try:
            produced = self.inner.rollup(parent, children, weights=weights, scores=scores)
        except Exception as error:  # noqa: BLE001 - a plugin's failure is a violation, not a crash
            failure = error
            produced = None
            violations = [self._violation(
                "C011", f"raised {type(error).__name__}: {error}", parent,
            )]
        else:
            produced, unreadable = self._read_once(produced)
            violations = unreadable or self._check(parent, children, produced)
        if not violations:
            return produced
        return self._resolve(parent, children, weights, scores, violations, failure)

    # -- checks ----------------------------------------------------------------
    def _read_once(self, produced: Any) -> tuple[Any, list[GuardViolation]]:
        """The output as exact values, read once: what is checked is what is returned."""
        if not isinstance(produced, ScorerOutput):
            return produced, []
        try:
            amount = produced.amount
            if isinstance(amount, (int, float)) and not isinstance(amount, bool):
                amount = float.__float__(amount) if isinstance(amount, float) else float(int.__int__(amount))
            void_rules = tuple(sorted({str(rule) for rule in list(produced.void_rules)}))
            frozen = ScorerOutput(
                kind=produced.kind, amount=amount,
                breakdown=dict(produced.breakdown) if isinstance(produced.breakdown, Mapping) else {},
                rules=tuple(str(rule) for rule in list(produced.rules)),
                mutations=tuple(produced.mutations or ()),
                metadata=dict(produced.metadata) if isinstance(produced.metadata, Mapping) else {},
                void_rules=void_rules,
            )
        except Exception as error:  # noqa: BLE001 - an unreadable output is a violation
            return produced, [self._violation(
                "C011", f"the output cannot be read: {type(error).__name__}: {error}", "?")]
        return frozen, []

    def _violation(self, code: str, message: str, parent: str) -> GuardViolation:
        return GuardViolation(
            code=code, stage=f"rollup:{self.label}", kind="hierarchy",
            message=f"{message} (parent {parent!r})",
        )

    def _check(
        self, parent: str, children: Sequence[ScorerOutput], produced: Any,
    ) -> list[GuardViolation]:
        if not isinstance(produced, ScorerOutput):
            return [self._violation(
                "C011", f"returned {type(produced).__name__}; a rollup returns a ScorerOutput", parent,
            )]
        found: list[GuardViolation] = []
        if _finite_number(produced.amount) is None:
            found.append(self._violation(
                "C011", f"amount {produced.amount!r} is not a finite number", parent,
            ))
        required = {rule for child in children for rule in child.void_rules}
        missing = sorted(required - set(produced.void_rules))
        if missing:
            found.append(self._violation(
                "C010",
                f"void_rules is missing {missing} reported by the children; a rollup must carry "
                "every child's void set upward",
                parent,
            ))
        if produced.mutations:
            found.append(self._violation(
                "C012", "emitted warning mutations; only the warning stage may", parent,
            ))
        return found

    def _resolve(
        self,
        parent: str,
        children: Sequence[ScorerOutput],
        weights: Mapping[str, float] | None,
        scores: Mapping[str, float] | None,
        violations: list[GuardViolation],
        failure: Exception | None,
    ) -> ScorerOutput:
        report = GuardReport(
            mode=self.mode,
            violations=tuple(violations),
            substituted=(self.label,) if self.mode == "fallback" else (),
        )
        self.reports.append(report)
        if self.mode == "enforce":
            raise ComplianceGuardViolation([item.as_dict() for item in violations]) from failure
        log.warning(
            "rollup %s violated %s at %s; using the reference kappa rollup",
            self.label, [item.code for item in violations], parent,
        )
        return self._reference.rollup(parent, children, weights=weights, scores=scores)

    # -- reporting ------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        inner = self.inner.describe() if callable(getattr(self.inner, "describe", None)) else {}
        return {
            "type": "guarded", "guard": "rollup", "mode": self.mode,
            "name": self.name, "inner": inner,
        }

    def __repr__(self) -> str:
        return f"<RollupGuard {self.label} mode={self.mode}>"


def guard_rollup(instance: Any, *, mode: str = "enforce", include_explicit: bool = True) -> Any:
    """Return ``instance`` unchanged if it is trusted, else wrapped in a ``RollupGuard``.

    ``None`` stays ``None`` (the caller reads it as "use the default"); an existing guard is
    not wrapped twice. ``include_explicit=False`` is the legacy ``build_rollup`` behaviour,
    where an in-code ``ExplicitRollup`` is the documented exception.
    """
    if instance is None or isinstance(instance, RollupGuard):
        return instance
    trusted = UNION_BY_CONSTRUCTION + (() if include_explicit else (ExplicitRollup,))
    if any(type(instance) is built for built in trusted):  # exact type: a subclass may override rollup()
        return instance
    return RollupGuard(instance, mode=mode)
