"""Reward terms and the stage that composes them (document 17 section 6.3)."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterable, Mapping, Sequence

from ..components import HUB
from ..exceptions import ComplianceError, ComponentError, ImproperlyConfigured, ValidationError
from ..rules.catalog import REWARDS
from .base import ScoreContext, ScorerDiagnostic, ScorerOutput, ScoringKind, severity_fraction
from .ledger import RuleEvent
from .reward_scorer import RewardScorer

#: Every reward rule id: the default ``rule_ids`` of a term that applies everywhere.
REWARD_IDS: frozenset[str] = frozenset(rule.id for rule in REWARDS)


class RewardTerm(ABC):
    """One factor of a reward: ``contribution(event, ctx) -> float >= 0``.

    ``rule_ids`` names the reward rules the term applies to. The contract (3.7) needs
    a non-empty ``rule_ids`` and a ``contribution`` method; subclassing this class is a
    convenience, not a requirement.
    """

    #: Registry-facing name; the decorator's name wins when they differ.
    name: ClassVar[str] = ""
    rule_ids: ClassVar[frozenset[str]] = REWARD_IDS

    @abstractmethod
    def contribution(self, event: RuleEvent, ctx: ScoreContext) -> float:
        """The event's contribution before the term's weight. Finite and non-negative."""

    def describe(self) -> dict[str, Any]:
        return {"name": self.name or type(self).__name__, "rule_ids": sorted(self.rule_ids)}


def _severity_table(ctx: ScoreContext) -> Mapping[str, Any]:
    return ctx.severity_weights or ctx.config.get("SEVERITY_WEIGHTS") or {}


def _rules(rules: Iterable[str] | None, *, owner: str) -> frozenset[str]:
    if rules is None:
        return REWARD_IDS
    chosen = frozenset(str(rule) for rule in rules)
    unknown = sorted(chosen - REWARD_IDS)
    if unknown:
        raise ValidationError(
            f"{owner}: {unknown} are not reward rules; reward terms apply to {sorted(REWARD_IDS)}"
        )
    if not chosen:
        raise ValidationError(f"{owner}: 'rules' is empty; omit it to apply to every reward rule")
    return chosen


class SeverityScaledTerm(RewardTerm):
    """``magnitude * severity_fraction``: the term the built-in ``RewardScorer`` embodies.

    ``composed_reward`` with this one term at weight 1 returns the same breakdown as
    ``RewardScorer`` for every ledger (test ``test_composed_matches_builtin``).
    """

    name = "severity_scaled"

    def __init__(self, *, rules: Sequence[str] | None = None) -> None:
        self.rule_ids = _rules(rules, owner="severity_scaled")

    def contribution(self, event: RuleEvent, ctx: ScoreContext) -> float:
        scale = severity_fraction(event.resolved_severity(), _severity_table(ctx), default=1.0)
        return event.magnitude * scale


class CappedTerm(RewardTerm):
    """``min(magnitude, limit) * severity_fraction``: stops one noisy rule dominating."""

    name = "capped"

    def __init__(self, *, limit: float, rules: Sequence[str] | None = None) -> None:
        number = float(limit)
        if not math.isfinite(number) or number < 0.0:
            raise ValidationError(f"capped: 'limit' must be finite and >= 0, got {limit!r}")
        self.limit = number
        self.rule_ids = _rules(rules, owner="capped")

    def contribution(self, event: RuleEvent, ctx: ScoreContext) -> float:
        scale = severity_fraction(event.resolved_severity(), _severity_table(ctx), default=1.0)
        return min(event.magnitude, self.limit) * scale

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "limit": self.limit}


class ComposedRewardScorer(RewardScorer):
    """A reward stage whose contribution per event is a weighted sum of terms.

    ``terms`` is a list of ``{type | import, params, weight}`` mappings (or built
    ``RewardTerm`` instances). The weight multiplies the term's contribution and must
    be finite and non-negative: a negative weight would let a reward event reduce a
    score, which the reward stage may not do (``reward_scorer.py:120-131``).

    Output shape is the ``RewardScorer`` shape plus ``terms`` (per-term totals), because
    ``ScoringPipeline._compose`` reads ``r_task``, ``r_epi``, ``r_collab``, ``r_align``,
    ``weights`` and ``by_rule`` and nothing else (``pipeline.py:511-523``).
    """

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        terms: Sequence[Any] | None = None,
        admitted_only: bool = False,
        trusted: bool = False,
    ) -> None:
        super().__init__(config=config)
        self.admitted_only = bool(admitted_only)
        self._trusted = bool(trusted)
        self.terms: list[tuple[str, float, Any]] = self._build_terms(
            [{"type": "severity_scaled"}] if terms is None else list(terms)
        )
        if not self.terms:
            raise ImproperlyConfigured("composed_reward needs at least one term")

    # -- construction ----------------------------------------------------
    def _build_terms(self, raw_terms: Sequence[Any]) -> list[tuple[str, float, Any]]:
        from ..components.resolver import ComponentResolver, ComponentSpec

        resolver = ComponentResolver()
        built: list[tuple[str, float, Any]] = []
        seen: dict[str, int] = {}
        #: ``[(label, Resolved)]`` for the terms built from a spec or adopted from code: what
        #: a checkpoint's component manifest records (document 17 section 14).
        self.term_records: list[tuple[str, Any]] = []
        for index, raw in enumerate(raw_terms):
            weight = 1.0
            if isinstance(raw, Mapping):
                entry = dict(raw)
                weight = self._weight(entry.pop("weight", 1.0), index)
                resolved = resolver.resolve(
                    ComponentSpec.parse("reward_term", entry), trusted=self._trusted
                )
                term = resolved.instance
            else:
                resolved = None
                term = raw
                if not callable(getattr(term, "contribution", None)) or not getattr(
                    term, "rule_ids", None
                ):
                    raise ImproperlyConfigured(
                        f"composed_reward term #{index} is neither a spec nor a RewardTerm"
                    )
            label = str(getattr(term, "name", "") or type(term).__name__)
            count = seen.get(label, 0)
            seen[label] = count + 1
            unique = label if count == 0 else f"{label}#{count + 1}"
            if resolved is None:
                try:
                    resolved = resolver.adopt("reward_term", term, name=label)
                except ComponentError:   # a duck-typed term the contract would refuse still runs
                    resolved = None
            if resolved is not None:
                self.term_records.append((unique, resolved))
            built.append((unique, weight, term))
        return built

    @staticmethod
    def _weight(value: Any, index: int) -> float:
        try:
            weight = float(value)
        except (TypeError, ValueError):
            raise ImproperlyConfigured(
                f"composed_reward term #{index}: weight {value!r} is not numeric"
            ) from None
        if not math.isfinite(weight) or weight < 0.0:
            raise ImproperlyConfigured(
                f"composed_reward term #{index}: weight must be finite and >= 0, got {weight!r}"
            )
        return weight

    # -- scoring ---------------------------------------------------------
    def score(self, ctx: ScoreContext) -> ScorerOutput:
        weights = self.bucket_weights()
        buckets = dict.fromkeys(self.BUCKETS, 0.0)
        by_rule: dict[str, float] = {}
        term_totals = {label: 0.0 for label, _weight, _term in self.terms}
        fired: list[str] = []
        ignored: list[str] = []
        skipped: list[str] = []
        allowed = ctx.allowed_rules if self.admitted_only else None

        for event in ctx.ledger.rewards():
            bucket = event.rule.bucket
            if bucket not in buckets:
                ignored.append(event.rule_id)
                continue
            if allowed is not None and event.rule_id not in allowed:
                skipped.append(event.rule_id)
                continue
            paid = 0.0
            matched = False
            for label, weight, term in self.terms:
                if event.rule_id not in term.rule_ids:
                    continue
                matched = True
                value = float(term.contribution(event, ctx))
                if not math.isfinite(value) or value < 0.0:
                    raise ComplianceError(
                        f"reward term {label!r} returned {value!r} for {event.rule_id}; a reward "
                        "contribution is finite and non-negative"
                    )
                amount = weight * value
                term_totals[label] += amount
                paid += amount
            if not matched:
                continue
            buckets[bucket] += paid
            by_rule[event.rule_id] = by_rule.get(event.rule_id, 0.0) + weights[bucket] * paid
            fired.append(event.rule_id)

        total = sum(weights[bucket] * buckets[bucket] for bucket in self.BUCKETS)
        rules = tuple(dict.fromkeys(fired))
        breakdown = {
            **buckets,
            "weights": dict(weights),
            "rules": list(rules),
            "by_rule": dict(by_rule),
            "ignored_rules": list(dict.fromkeys(ignored)),
            "total": total,
            "terms": dict(term_totals),
        }
        return ScorerOutput(
            kind=self.kind,
            amount=total,
            breakdown=breakdown,
            rules=rules,
            metadata={
                "ignored_rules": tuple(dict.fromkeys(ignored)),
                "skipped_not_admitted": tuple(dict.fromkeys(skipped)),
                "terms": [label for label, _weight, _term in self.terms],
            },
        )

    # -- introspection ---------------------------------------------------
    def validate(self, settings: Mapping[str, Any]) -> list[ScorerDiagnostic]:
        diagnostics = super().validate(settings)
        covered: set[str] = set()
        for _label, _weight, term in self.terms:
            covered |= set(term.rule_ids)
        if not covered:
            diagnostics.append(
                ScorerDiagnostic(
                    level="error",
                    code="rewards.no_rules_covered",
                    message="no term applies to any reward rule, so every reward would be zero",
                )
            )
        if all(weight == 0.0 for _label, weight, _term in self.terms):
            diagnostics.append(
                ScorerDiagnostic(
                    level="warning",
                    code="rewards.all_terms_zero_weight",
                    message="every term has weight 0, so every reward is zero",
                    hint="use the 'zero_reward' stage to say so on purpose",
                )
            )
        return diagnostics

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload["terms"] = [
            {"label": label, "weight": weight, **_describe_term(term)}
            for label, weight, term in self.terms
        ]
        payload["admitted_only"] = self.admitted_only
        return payload


def _describe_term(term: Any) -> dict[str, Any]:
    """``describe()`` when the term has one, else a derived description. A term that
    satisfies the contract without subclassing ``RewardTerm`` need not define it."""
    describe = getattr(term, "describe", None)
    if callable(describe):
        return dict(describe())
    return {"name": type(term).__name__, "rule_ids": sorted(term.rule_ids)}


# Built-ins (origin ``builtin``), registered the way the legacy tables register their own
# entries: they bypass locks and the per-kind cap, so a project that locks ``scorer:*`` can
# still import this module.
HUB.registry("reward_term").register_builtin("severity_scaled", SeverityScaledTerm)
HUB.registry("reward_term").register_builtin("capped", CappedTerm)
HUB.registry("scorer").register_builtin("composed_reward", ComposedRewardScorer)
