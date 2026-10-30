"""Builders shared by the scoring-component tests (document 17 section 6).

``stage_variant`` makes a stage that is *not* a built-in class (so the guard inspects it)
out of a built-in one plus a function over its output: the shape of every plausible bug and
every adversary in 6.4.1.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Callable, Mapping

from mactf import conf
from mactf.components.guards.compliance import ComplianceGuard, GuardPolicy
from mactf.scoring.base import ScoreContext, ScorerOutput, ScoringKind
from mactf.scoring.ledger import Ledger
from mactf.scoring.penalty import PenaltyScorer
from mactf.scoring.pipeline import ScoringPipeline, build_scorer
from mactf.scoring.reward_scorer import RewardScorer
from mactf.scoring.warning import WarningPolicy
from mactf.scoring.warning_scorer import WarningScorer

Fn = Callable[[ScorerOutput, ScoreContext], ScorerOutput]


def settings_copy(**blocks: Any) -> dict[str, Any]:
    """The framework defaults as a plain, private dict, with ``blocks`` merged key by key."""
    settings = copy.deepcopy(conf.DEFAULTS)
    for name, values in blocks.items():
        settings.setdefault(name, {})
        settings[name].update(values)
    return settings


def ledger_of(spec: Mapping[str, float] | None = None, *, peer: str = "a") -> Ledger:
    ledger = Ledger(peer=peer)
    for rule_id, magnitude in (spec or {}).items():
        ledger.record(rule_id, magnitude=magnitude)
    return ledger


def with_output(output: ScorerOutput, breakdown: Mapping[str, Any] | None = None, **changes: Any):
    """``output`` with fields replaced; ``breakdown`` is merged over the old one."""
    if breakdown is not None:
        changes["breakdown"] = {**dict(output.breakdown), **dict(breakdown)}
    return replace(output, **changes)


def stage_variant(base: type, fn: Fn | None = None, *, name: str = "Variant", **attrs: Any) -> type:
    """A subclass of ``base`` whose ``score`` post-processes the built-in output with ``fn``."""

    def score(self: Any, ctx: ScoreContext) -> ScorerOutput:
        output = base.score(self, ctx)
        return fn(output, ctx) if fn is not None else output

    return type(name, (base,), {"score": score, **attrs})


def penalty_variant(fn: Fn | None = None, **attrs: Any) -> type:
    return stage_variant(PenaltyScorer, fn, name="CustomPenalty", **attrs)


def reward_variant(fn: Fn | None = None, **attrs: Any) -> type:
    return stage_variant(RewardScorer, fn, name="CustomReward", **attrs)


def warning_variant(prepare: Callable[[Any, ScoreContext], ScoreContext], **attrs: Any) -> type:
    """A WARNING stage whose ``prepare`` runs the built-in one and then ``prepare``."""

    def run(self: Any, ctx: ScoreContext) -> ScoreContext:
        return prepare(self, WarningScorer.prepare(self, ctx))

    return type("CustomWarning", (WarningScorer,), {"prepare": run, **attrs})


def pipeline_with(
    *,
    penalty: Any = None,
    reward: Any = None,
    warning: Any = None,
    mode: str = "enforce",
    settings: Mapping[str, Any] | None = None,
    **guard_options: Any,
) -> ScoringPipeline:
    """A pipeline over the default settings with the given stage *classes or instances* swapped in."""
    base = dict(settings or settings_copy())
    score_config = dict(base["SCORING"])
    policy = WarningPolicy.from_settings(base["WARNINGS"])

    def make(choice: Any, builtin: type, **kwargs: Any) -> Any:
        if choice is None:
            return builtin(config=score_config, **kwargs)
        return choice(config=score_config, **kwargs) if isinstance(choice, type) else choice

    stages = [
        make(warning, WarningScorer, policy=policy),
        make(reward, RewardScorer),
        make(penalty, PenaltyScorer),
    ]
    return ScoringPipeline(
        stages,
        score_config=score_config,
        severity_weights=dict(base["SEVERITY"]["WEIGHTS"]),
        warning_policy=policy,
        guard=ComplianceGuard(GuardPolicy(mode=mode, **guard_options)),
    )


def default_pipeline(**overrides: Any) -> ScoringPipeline:
    return build_scorer(settings_copy(**overrides))


KINDS = {"warning": ScoringKind.WARNING, "reward": ScoringKind.REWARD, "penalty": ScoringKind.PENALTY}
