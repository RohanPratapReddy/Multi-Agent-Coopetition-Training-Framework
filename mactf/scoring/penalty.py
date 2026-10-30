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
        configured = self.config.get("NON_OFFSETTABLE")
        if configured is None:
            return NON_OFFSETTABLE
        return frozenset(configured)

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
