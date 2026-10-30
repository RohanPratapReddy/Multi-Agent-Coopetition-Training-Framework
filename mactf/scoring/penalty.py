"""The penalty stage: ``P = (sum_k lambda_k * p_k) * Omega``.

This is :func:`mactf.scoring.continuous._penalty_breakdown` with a public name.
Three details decide whether the number matches, and all three are load-bearing:

1. **Lambda resolution** is per rule with ``DEFAULT_LAMBDA`` as fallback
   (``continuous.py:145-157``).
2. **The warning-only exclusion** is mandatory. framework.md W1 makes a warning
   zero-score, so a rule that produced a warning this episode and did not
   escalate must not also be subtracted. The exclusion set arrives already
   computed, as ``ctx.warning_events - ctx.escalated_rules``; this class never
   reads the registry.
3. **Omega is a bounded product.** The docx amplifier is unbounded; each factor
   is clamped with ``max(1.0, factor)`` so the product can never attenuate. This
   is current behaviour and it is preserved deliberately.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..compliance import DEFAULT_MIN_FLOOR_LAMBDA, effective_non_offsettable
from ..exceptions import ComplianceError, ImproperlyConfigured
from ..rules.catalog import NON_OFFSETTABLE
from .base import ScoreContext, Scorer, ScorerDiagnostic, ScorerOutput, ScoringKind

#: ``continuous.py``'s default when ``SCORING.DEFAULT_LAMBDA`` is absent.
DEFAULT_LAMBDA = 1.0
#: ``continuous.py``'s default Omega factor when ``SANDBOX_MULTIPLIER`` is absent.
DEFAULT_SANDBOX_MULTIPLIER = 1000.0


class PenaltyScorer(Scorer):
    """Scores penalty events with lambda weights and the bounded Omega product."""

    kind = ScoringKind.PENALTY
    order = 20
    requires = frozenset({"ledger", "config", "escalated_rules"})

    # -- config ----------------------------------------------------------
    @property
    def lambdas(self) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in (self.config.get("PENALTY_LAMBDAS") or {}).items()
        }

    @property
    def default_lambda(self) -> float:
        return float(self.config.get("DEFAULT_LAMBDA", DEFAULT_LAMBDA))

    @property
    def non_offsettable(self) -> frozenset[str]:
        return effective_non_offsettable(self.config.get("NON_OFFSETTABLE"))

    @property
    def min_floor_lambda(self) -> float:
        return float(self.config.get("MIN_FLOOR_LAMBDA", DEFAULT_MIN_FLOOR_LAMBDA))

    def _allowed(self, ctx: ScoreContext) -> frozenset[str] | None:
        """The rules the controlling tier admits.

        In ``strict`` mode the pipeline injects the controlling tier's rule ids.
        In ``layered`` mode the pipeline injects nothing and everything scores.
        """
        mode = str(ctx.config.get("PRECEDENCE_MODE", self.config.get("PRECEDENCE_MODE", "strict")))
        if mode != "strict":
            return None
        return ctx.allowed_rules

    # -- scoring ---------------------------------------------------------
    def score(self, ctx: ScoreContext) -> ScorerOutput:
        lambdas = self.lambdas
        default_lambda = self.default_lambda
        allowed = self._allowed(ctx)
        warning_only = ctx.warning_only_rules
        non_offsettable = self.non_offsettable
        floor_lambda = self.min_floor_lambda

        amounts: dict[str, float] = {}
        excluded: list[str] = []
        for event in ctx.ledger.penalties():
            if event.magnitude < 0.0:
                # Ledger.record rejects this, so reaching here means a custom
                # ledger implementation admitted it. Fail rather than pay a peer.
                raise ComplianceError(
                    f"{event.rule_id} carries a negative magnitude "
                    f"({event.magnitude!r}); a penalty cannot be a reward"
                )
            if allowed is not None and event.rule_id not in allowed:
                continue
            if event.rule_id in warning_only:
                excluded.append(event.rule_id)
                continue
            lam = float(lambdas.get(event.rule_id, default_lambda))
            if event.rule_id in non_offsettable:
                lam = max(lam, floor_lambda)  # H2: a floor can never be configured to zero
            amounts[event.rule_id] = amounts.get(event.rule_id, 0.0) + lam * event.magnitude

        omega, omega_rules = self._omega(amounts)
        raw_total = sum(amounts.values())
        total = raw_total * omega
        rules = tuple(sorted(amounts))
        void_rules = tuple(sorted(self.non_offsettable & set(amounts)))

        breakdown = {
            "amounts": dict(amounts),
            "raw_total": raw_total,
            "omega": omega,
            "omega_rules": list(omega_rules),
            "total": total,
            "rules": list(rules),
        }
        return ScorerOutput(
            kind=self.kind,
            amount=total,
            breakdown=breakdown,
            rules=rules,
            metadata={
                "warning_only_excluded": tuple(dict.fromkeys(excluded)),
                "raw_total": raw_total,
                "omega": omega,
            },
            void_rules=void_rules,
        )

    def _omega(self, amounts: Mapping[str, float]) -> tuple[float, tuple[str, ...]]:
        omega_rules = set(self.config.get("OMEGA_RULES") or [])
        overrides = self.config.get("OMEGA_RULE_OVERRIDES") or {}
        base = float(self.config.get("SANDBOX_MULTIPLIER", DEFAULT_SANDBOX_MULTIPLIER))
        triggered = [rule_id for rule_id in amounts if rule_id in omega_rules]
        if not triggered:
            return 1.0, ()
        multiplier = 1.0
        for rule_id in triggered:
            factor = float(overrides.get(rule_id, base))
            multiplier *= max(1.0, factor)
        return multiplier, tuple(sorted(triggered))

    # -- introspection ---------------------------------------------------
    def validate(self, settings: Mapping[str, Any]) -> list[ScorerDiagnostic]:
        diagnostics: list[ScorerDiagnostic] = []
        raw = settings.get("PENALTY_LAMBDAS", self.config.get("PENALTY_LAMBDAS")) or {}
        for rule_id, value in raw.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                diagnostics.append(
                    ScorerDiagnostic(
                        level="error",
                        code="penalties.non_numeric_lambda",
                        message=f"PENALTY_LAMBDAS[{rule_id!r}] is not numeric: {value!r}",
                    )
                )
                continue
            if number < 0.0:
                # A negative lambda would turn a violation into a bonus.
                diagnostics.append(
                    ScorerDiagnostic(
                        level="error",
                        code="penalties.negative_lambda",
                        message=f"PENALTY_LAMBDAS[{rule_id!r}] is negative: {number!r}",
                        hint="penalty lambdas must be non-negative",
                    )
                )
        multiplier = settings.get(
            "SANDBOX_MULTIPLIER", self.config.get("SANDBOX_MULTIPLIER", DEFAULT_SANDBOX_MULTIPLIER)
        )
        try:
            if float(multiplier) < 1.0:
                diagnostics.append(
                    ScorerDiagnostic(
                        level="error",
                        code="penalties.omega_below_one",
                        message=(
                            "SANDBOX_MULTIPLIER is below 1.0, which would attenuate a "
                            "penalty; Omega is an amplifier only"
                        ),
                    )
                )
        except (TypeError, ValueError):
            diagnostics.append(
                ScorerDiagnostic(
                    level="error",
                    code="penalties.non_numeric_omega",
                    message=f"SANDBOX_MULTIPLIER is not numeric: {multiplier!r}",
                )
            )

        configured = settings.get("NON_OFFSETTABLE", self.config.get("NON_OFFSETTABLE"))
        if configured is not None:
            narrowed = sorted(NON_OFFSETTABLE - frozenset(configured))
            if narrowed:
                diagnostics.append(
                    ScorerDiagnostic(
                        level="error",
                        code="penalties.non_offsettable_narrowed",
                        message=f"NON_OFFSETTABLE omits {narrowed}; it may only add rules",
                        hint="remove the key to use the catalog set, or list a superset of it",
                    )
                )
        try:
            floor = float(
                settings.get(
                    "MIN_FLOOR_LAMBDA",
                    self.config.get("MIN_FLOOR_LAMBDA", DEFAULT_MIN_FLOOR_LAMBDA),
                )
            )
            default = float(settings.get("DEFAULT_LAMBDA", self.default_lambda))
            for rule_id in sorted(effective_non_offsettable(configured)):
                try:
                    effective = float((raw or {}).get(rule_id, default))
                except (TypeError, ValueError):
                    continue  # already reported as penalties.non_numeric_lambda
                if effective < 0.0:
                    continue  # already reported as penalties.negative_lambda
                if effective < floor:
                    diagnostics.append(
                        ScorerDiagnostic(
                            level="error",
                            code="penalties.floor_lambda_below_minimum",
                            message=(
                                f"PENALTY_LAMBDAS[{rule_id!r}] resolves to {effective:g}, below "
                                f"MIN_FLOOR_LAMBDA {floor:g}; a non-offsettable floor cannot be "
                                "weakened"
                            ),
                            hint=f"raise the lambda to at least {floor:g}",
                        )
                    )
        except (TypeError, ValueError):
            diagnostics.append(
                ScorerDiagnostic(
                    level="error",
                    code="penalties.non_numeric_floor",
                    message="MIN_FLOOR_LAMBDA or DEFAULT_LAMBDA is not numeric",
                )
            )
        return diagnostics

    def assert_valid(self, settings: Mapping[str, Any] | None = None) -> None:
        problems = [
            diagnostic
            for diagnostic in self.validate(settings or self.config)
            if diagnostic.level == "error"
        ]
        if problems:
            raise ImproperlyConfigured(
                "; ".join(f"{item.code}: {item.message}" for item in problems)
            )
