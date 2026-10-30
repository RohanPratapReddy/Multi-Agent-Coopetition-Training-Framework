---
title: Scorers and Function Wrappers
status: approved-for-implementation
version: 1.0.0
applies_to: mactf >= 0.1.0
owner: TBD
last_reviewed: 2026-10-20
---

# 07. Scorers and Function Wrappers

This document fixes the object model for scoring and for the functional pieces of
the training objective. Two things change relative to the current library:

1. `Scorer` stops being one concrete dataclass that does everything and becomes
   an abstract base class with three child scorers: `RewardScorer`,
   `PenaltyScorer`, `WarningScorer`. The concrete composite is
   `ScoringPipeline(Scorer)`.
2. The free functions in `mactf/rl/` gain a wrapper layer: `ValueFunction`,
   `AdvantageFunction`, `LossFunction`, and the supporting wrappers named in
   section 8. The existing free functions keep their signatures and remain the
   implementation; the wrappers are the composition surface.

The arithmetic does not change. This is a refactor plus an interface layer, and
section 11 makes numerical equivalence a hard gate. Nothing here is
re-derived from `docs/framework.md`; that document stays normative for scoring
semantics and this one stays normative for interface shape.

## 1. Why the split is worth doing

Today `Scorer` (`mactf/scoring/pipeline.py:30`) is a dataclass with four fields
that performs five distinct jobs in `score_peer`:

- it advances the warning registry (`pipeline.py:73-88`),
- it computes the reward (`pipeline.py:95-101` via `continuous_score`),
- it computes the penalty including Omega (`continuous.py:137-176`),
- it applies precedence and the non-offsettable floor (`pipeline.py:103-104`),
- it emits signals and builds the scorecard (`pipeline.py:138-139`).

Each job has a different failure mode and a different reason to be replaced. A
user who only wants a different reward bucket weighting must not have to
subclass a five-job object. A user who wants energy-based or shape-based
rewards for a vision model must be able to supply a `RewardScorer` and keep the
compliant penalty and warning machinery untouched. Splitting the class makes
those extensions additive and keeps the compliance floor in one place that
cannot be accidentally overridden.

The split also gives hierarchical scoring (document 10) the seam it needs: a
`HierarchicalScorer` can compose per-node `ScoringPipeline` instances and roll
their outputs up without reimplementing any equation.

## 2. Module layout

```
mactf/scoring/
  base.py             NEW  Scorer (ABC), ScoringKind, ScoreContext, ScorerOutput,
                           WarningMutation, ScorerDiagnostic
  reward_scorer.py    NEW  RewardScorer
  penalty.py          NEW  PenaltyScorer        (note: file name is penalty.py,
                           class name is PenaltyScorer, matching the plan)
  warning_scorer.py   NEW  WarningScorer
  pipeline.py         KEPT Scorer legacy shim, ScoringPipeline, build_scorer
  hierarchy.py        NEW  HierarchicalScorer, RollupStrategy and implementations
  points.py           NEW  MAX_POINTS, SeverityPoints, PointsTable, DEFAULT table
  evidence.py         NEW  EpisodeEvidence, ClaimRecord, ClaimAudit
  guards.py           NEW  RuleBinding, GuardVerdict, the 40 guards, registries
  normalize.py        NEW  NormalizationPolicy, CohortStats, CohortNormalizer
  continuous.py       KEPT unchanged; the equation functions
  warning.py          KEPT unchanged; WarningRegistry and WarningPolicy
  ledger.py           KEPT unchanged
  scorecard.py        KEPT unchanged
  __init__.py         UPDATED exports
```

There is no `penalty_scorer.py`. The plan text says `mactf/scoring/penalty.py`
and that is what ships; the class inside it is `PenaltyScorer`. This is stated
explicitly because a mismatched file name is the kind of detail that produces a
missing import in phase A0.

`points.py`, `evidence.py`, `guards.py`, and `normalize.py` are specified in
[11-points-and-normalization.md](11-points-and-normalization.md). They sit in
`mactf/scoring/` rather than a new package because they are scoring concerns,
they depend only on `mactf/rules/` and the ledger, and putting them elsewhere
would make the bay-layer rule in
[01-target-architecture.md](01-target-architecture.md) section 4 read as
"except these four". Direction of dependency is strictly one-way:

```
points.py   <- evidence.py <- guards.py <- reward_scorer.py / penalty.py
points.py   <- normalize.py <- pipeline.py
points.py   -/-> continuous.py        (no import in either direction)
```

The last line is enforced by a test: `continuous.py` must not import `points.py`,
and `points.py` must not import `continuous.py`. Both are pure and both are
imported by the pipeline, but they must not see each other, because the whole
equivalence argument rests on neither being able to affect the other.

## 3. `Scorer` base class

```python
# mactf/scoring/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, ClassVar, Mapping, Sequence

from ..rules.severity import Severity
from ..scoring.ledger import Ledger
from ..scoring.warning import WarningPolicy


class ScoringKind(str, Enum):
    REWARD = "reward"
    PENALTY = "penalty"
    WARNING = "warning"
    PIPELINE = "pipeline"
    HIERARCHY = "hierarchy"


@dataclass(frozen=True)
class WarningMutation:
    """One state transition a WarningScorer wants applied to a registry."""

    peer: str
    rule_id: str
    episode: int
    action: str            # "issue" | "escalate" | "expire" | "clean"
    severity: Severity | None = None
    evidence: str = ""
    penalty: float = 0.0


@dataclass(frozen=True)
class ScoreContext:
    """Everything a scorer may read. Frozen so a child cannot mutate its input."""

    peer: str
    episode: int
    ledger: Ledger
    node_path: str = ""
    depth: int = 0
    config: Mapping[str, Any] = field(default_factory=dict)
    severity_weights: Mapping[str, Any] = field(default_factory=dict)
    warning_policy: WarningPolicy = field(default_factory=WarningPolicy)
    known_rules: frozenset[str] = frozenset()
    warning_total: float = 0.0
    warning_events: tuple[str, ...] = ()
    escalated_rules: frozenset[str] = frozenset()
    children: tuple["ScorerOutput", ...] = ()
    branch_void: frozenset[str] = frozenset()
    extras: Mapping[str, Any] = field(default_factory=dict)

    def evolve(self, **changes: Any) -> "ScoreContext":
        return replace(self, **changes)


@dataclass(frozen=True)
class ScorerOutput:
    """The result of one scorer stage."""

    kind: ScoringKind
    amount: float = 0.0
    breakdown: Mapping[str, Any] = field(default_factory=dict)
    rules: tuple[str, ...] = ()
    mutations: tuple[WarningMutation, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    void_rules: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "amount": self.amount,
            "breakdown": dict(self.breakdown),
            "rules": list(self.rules),
            "metadata": dict(self.metadata),
            "void_rules": list(self.void_rules),
        }


@dataclass(frozen=True)
class ScorerDiagnostic:
    level: str             # "info" | "warning" | "error"
    code: str
    message: str
    hint: str = ""


class Scorer(ABC):
    """Base class for every scoring stage.

    Subclasses implement exactly one job. The pipeline owns ordering, the
    registry, compliance, and the scorecard.
    """

    kind: ClassVar[ScoringKind]
    order: ClassVar[int] = 100
    requires: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    # -- lifecycle -------------------------------------------------------
    def prepare(self, ctx: ScoreContext) -> ScoreContext:
        """Return the context the following stages will see.

        Only :class:`WarningScorer` overrides this, because only warning
        detection can change what later stages must know.
        """
        return ctx

    @abstractmethod
    def score(self, ctx: ScoreContext) -> ScorerOutput:
        """Compute this stage's contribution."""

    def finalize(self, ctx: ScoreContext, output: ScorerOutput) -> ScorerOutput:
        """Post-process an output. Default is a no-op."""
        return output

    # -- introspection ---------------------------------------------------
    def validate(self, settings: Mapping[str, Any]) -> list[ScorerDiagnostic]:
        return []

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "class": type(self).__name__,
            "order": self.order,
            "requires": sorted(self.requires),
            "config_keys": sorted(self.config),
        }

    def state_dict(self) -> dict[str, Any]:
        return {"class": type(self).__name__, "config": dict(self.config)}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if payload.get("class") not in (None, type(self).__name__):
            raise ValueError(
                f"{type(self).__name__} cannot load state written by "
                f"{payload.get('class')!r}"
            )
        self.config = dict(payload.get("config", {}))
```

Design points that the implementation must honour:

- `ScoreContext` is frozen. A stage returns a new context through `evolve()`
  rather than mutating. This is what makes ordering explicit instead of
  incidental.
- `Scorer.kind`, `order`, and `requires` are class variables, so a chain can be
  validated without instantiating anything expensive.
- `prepare` is separate from `score` because warning detection must run before
  penalty computation. Merging them into one method is what produced the
  ordering hazard described in section 6.
- No stage touches `PeerScorecard`. Only `ScoringPipeline` builds one.

## 4. `RewardScorer`

```python
# mactf/scoring/reward_scorer.py
class RewardScorer(Scorer):
    kind = ScoringKind.REWARD
    order = 10
    requires = frozenset({"ledger", "config"})

    BUCKETS = ("r_task", "r_epi", "r_collab", "r_align")

    def score(self, ctx: ScoreContext) -> ScorerOutput: ...
```

Rules:

- It reads `ctx.ledger.rewards()` and only reward rules. It never looks at
  `ledger.penalties()`.
- Bucket weights come from `config["REWARD_WEIGHTS"]` with the same defaults
  `continuous.py:121` uses today.
- Severity scales a reward inside its band with
  `float(SEVERITY_WEIGHTS[severity.label])`, defaulting to `1.0` when a label is
  absent, exactly as `continuous.py:130-132` does. A reward at `S0` has weight
  `0.0` in the default table and therefore contributes nothing.
- Rules whose bucket is not a `RewardBreakdown` field are skipped and recorded in
  `breakdown["ignored_rules"]` rather than silently dropped, matching
  `continuous.py:126-127`.
- The output amount is `reward.total`, the same number
  `RewardBreakdown.total` (`continuous.py:44-48`) produces. Reward is a
  non-negative quantity; a negative configured weight is rejected by
  `validate()`.
- `breakdown` carries all four buckets, their weights, the contributing rule
  ids in first-seen order, and `ignored_rules`.
- Precedence suppression is *not* this class's job. It reports the raw reward;
  `ScoringPipeline` applies `rules.precedence.resolve` and zeroes the reward
  when the controlling tier has no reward rule, exactly as
  `continuous.py:217-219` does today. Putting suppression in `RewardScorer`
  would mean a custom reward scorer could bypass tier precedence.

Extension contract: a subclass may replace `score()` entirely. The only
guarantees it must preserve are (a) the return type, (b) non-negative weight
validation, and (c) not reading warning state. A vision task that scores image
similarity instead of classification correctness subclasses this and nothing
else changes.

### 4.1 Points projection

`RewardScorer` produces **two** numbers per peer, not one: `reward.total` (the
docx quantity, unchanged) and `reward_points` (the 0-100 rubric quantity).
[11-points-and-normalization.md](11-points-and-normalization.md) section 3 is the
table; this subsection is how the class uses it.

```python
# mactf/scoring/reward_scorer.py
    def score(self, ctx: ScoreContext) -> ScorerOutput: ...

    def points(self, ctx: ScoreContext, output: ScorerOutput) -> RewardPoints: ...
```

The `points()` method is separate from `score()` on purpose:

- `score()` must keep producing the number the evidence gate in section 11
  compares against `_reward_breakdown`. If points were computed inside it, a
  points-table mistake could perturb the docx total and the gate would have
  nothing stable to compare against.
- `points()` reads the same ledger and the same `ctx`, so it cannot disagree about
  *which* rules fired. It can disagree about whether a reward was *earned*,
  because guards run here and do not run in the docx path (section 4.3).

Arithmetic, per earned reward event:

```
rule_points = DEFAULT_REWARD_WEIGHTS[rule_id]
              * severity_fraction(severity)
              * 0.5 ** (occurrence_index - 1)
```

The first occurrence of a rule earns its full weight; the second earns half; the
third a quarter. Occurrence index is per `(peer, rule_id)` within the episode and
counts only **earned** occurrences, so a denied event does not consume
diminishing-returns budget. The geometric factor is what stops a peer from farming
R5 twelve times and reaching the cap on repetition alone. The docx path has no
such factor, which is one of the reasons the two numbers diverge, and section 10.2
of `11` explains why that divergence is correct.

Ordering rule: rewards are summed in rule-id sorted order, then rounded once, so
two peers with the same reward multiset get bit-identical points regardless of
ledger insertion order. `reward_points` is then clamped to `MAX_POINTS`.

### 4.2 Guards are not the reward scorer's decision

A reward is only projected when its guard in
[11-points-and-normalization.md](11-points-and-normalization.md) section 4 says it
was earned. `RewardScorer.points()` consults `ctx.extras["guard_registry"]`, looks
up each firing rule id, and for each `GuardVerdict`:

| Verdict | Effect |
|---|---|
| `earned=True` | full projection above |
| `earned=False, reason=...` | `0.0`, and a `DenialRecord(rule_id, reason, missing)` appended |
| `missing=(...)` | `0.0` with `reason="missing_evidence"`, and the field names recorded |

The guard registry is not editable from a scorer subclass. It arrives through
`ScoreContext.extras`, seeded by `ScoringPipeline`, exactly as `allowed_rules`
does for penalties. A custom reward scorer can *add* a guard for a custom rule id
it introduced, through `RulePoints.custom_guards`, but it cannot remove a
framework guard. That preserves the framework's "no credit without evidence"
property while keeping the extension contract honest: a vision scorer that adds
an `R18` owns that rule's guard.

Denials land in `ScorerOutput.metadata["points_denials"]`. They are surfaced by
`PointsBreakdown.denied` and by `GET /scores/scale`, so a peer that earned
nothing can be told exactly why rather than being handed a silent zero.

### 4.3 Why both numbers can legitimately differ

The docx path credits a reward when the ledger contains the event. The points
path credits it when the ledger contains the event **and** the guard passes. Three
cases produce a difference, all intentional:

| Situation | `reward.total` | `reward_points` |
|---|---|---|
| Reward event present, guard passes | contributes | full weight |
| Reward event present, guard denies (e.g. R1 alongside a P1) | contributes | `0.0` + denial |
| Reward event present, guard lacks evidence | contributes | `0.0` + `missing` |

The framework's own integrity rules say R1 must not be awarded when the team
fabricated; the docx equations as written have no way to express that, because
`R1` is a sum over events. The points path is where the framework's text is
actually enforced. Section 10.2 of `11` carries the sentence that must survive
review: the two numbers answer different questions.

## 5. `PenaltyScorer`

```python
# mactf/scoring/penalty.py
class PenaltyScorer(Scorer):
    kind = ScoringKind.PENALTY
    order = 20
    requires = frozenset({"ledger", "config", "escalated_rules"})

    def score(self, ctx: ScoreContext) -> ScorerOutput: ...
```

Rules:

- It reads `ledger.penalties()` only.
- Lambda resolution: `config["PENALTY_LAMBDAS"][rule_id]` falling back to
  `config["DEFAULT_LAMBDA"]`, matching `continuous.py:145-157`.
- The **warning-only exclusion** is mandatory and is why `escalated_rules` is in
  `requires`. A rule that produced a warning this episode and did not escalate
  contributes zero to `P` (framework.md W1). The exclusion set is
  `ctx.warning_events - ctx.escalated_rules`. This class must never read the
  registry; it receives the answer, it does not ask the question.
- Suppression by precedence: when `ctx.config["PRECEDENCE_MODE"] == "strict"`,
  only rules in the controlling tier contribute. When `"layered"`, all do. The
  allowed set arrives through `ctx.extras["allowed_rules"]`, computed by the
  pipeline, so a custom penalty scorer cannot invent a wider allow list.
- Omega: `config["OMEGA_RULES"]` triggers a multiplier that is the product of
  `config["OMEGA_RULE_OVERRIDES"]` values, defaulting to
  `config["SANDBOX_MULTIPLIER"]`. Each factor is clamped with
  `max(1.0, factor)` so Omega is bounded below by 1 and the docx's unbounded
  amplifier becomes a bounded product. This is the current behaviour at
  `continuous.py:159-170` and it is preserved deliberately.
- `breakdown` carries `amounts`, `raw_total`, `omega`, `omega_rules`,
  `total`, and `rules` (sorted), which is the shape
  `PenaltyBreakdown.as_dict()` already returns (`continuous.py:79-86`).
- `void_rules` is populated with the intersection of this episode's penalties
  and `NON_OFFSETTABLE`. The pipeline uses it for `branch_void` propagation in
  hierarchical mode.
- A negative magnitude is impossible: `Ledger.record` already rejects it. If a
  custom ledger implementation admits one, `validate()` is not the guard;
  `score()` raises `ComplianceError` rather than producing a reward.

### 5.1 Points projection

`PenaltyScorer` gains the mirror of `RewardScorer.points()`:

```python
# mactf/scoring/penalty.py
    def points(self, ctx: ScoreContext, output: ScorerOutput) -> PenaltyPoints: ...
```

Three properties distinguish it from the reward side, and all three come straight
from the framework:

1. **Linear accumulation, no diminishing returns.** A penalty at weight `w` with
   severity fraction `f` subtracts `w * f`. Firing twice subtracts twice. The
   framework's penalty ladder is absolute, and halving a repeated violation would
   create the exact farm the severity ladder is meant to close.
2. **Warning-only events subtract nothing.** The exclusion set is already computed
   for `score()`. `points()` reuses `ctx.warning_events - ctx.escalated_rules`
   rather than recomputing it, so the two numbers cannot disagree about which
   events were warning-only.
3. **Non-offsettable penalties are terminal.** When the firing set intersects
   `NON_OFFSETTABLE`, the penalty points still accumulate (for display and for
   `PointsBreakdown`), but `final_points` is pinned to `0.0` by the pipeline
   (section 7.2). The penalty scorer reports; the pipeline pins.

Severity fractions are the superlinear ones from
[11-points-and-normalization.md](11-points-and-normalization.md) section 2.3
(`0.00, 0.05, 0.15, 0.40, 1.00`), which is a different table from the linear docx
`SEVERITY_WEIGHTS`. That is deliberate: `continuous.py` keeps its linear weights
so `final_score` is unchanged, while the rubric escalates. The two tables share
nothing and must not be unified by a future refactor; doing so would change
`final_score`.

`penalty_points` is capped at `MAX_POINTS` before it reaches the pipeline, and the
cap is applied to the sum rather than per-rule so a long tail of small violations
still saturates. The saturation is recorded as
`PenaltyPoints(..., saturated=True)` so a report can say "capped" instead of
implying 100 points of genuine damage.

## 6. `WarningScorer`

```python
# mactf/scoring/warning_scorer.py
class WarningScorer(Scorer):
    kind = ScoringKind.WARNING
    order = 5          # before reward and penalty on purpose
    requires = frozenset({"ledger", "warning_policy"})

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        policy: WarningPolicy | None = None,
        registries: dict[str, WarningRegistry] | None = None,
    ) -> None: ...

    def prepare(self, ctx: ScoreContext) -> ScoreContext: ...
    def score(self, ctx: ScoreContext) -> ScorerOutput: ...
    def registry_for(self, peer: str) -> WarningRegistry: ...
    def registry_for_node(self, node_path: str, peer: str) -> WarningRegistry: ...
    def begin_episode(self, peer: str, *, episode: int, node_path: str = "") -> list[Any]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, payload: Mapping[str, Any]) -> None: ...
```

This is the class with real state, and its ordering is the one thing that must
not be negotiable.

**Ordering hazard, stated once.** Today `Scorer.score_peer` issues warnings
*first* (`pipeline.py:73-88`), collects `warning_events` and
`escalated_rules`, then calls `continuous_score` with both
(`pipeline.py:95-101`). If warnings were computed after penalties, a rule that
matured this episode would be scored both as a warning-only rule (excluded from
P) and as a penalty (included), or vice versa. Both are wrong. Therefore:

- `WarningScorer.order` is `5`. `RewardScorer` is `10`, `PenaltyScorer` is
  `20`. The pipeline sorts ascending by order.
- `prepare()` is the only place the registry advances. It returns a context
  whose `warning_total`, `warning_events`, and `escalated_rules` are already
  filled in.
- `score()` is read-only and reports the *same* numbers it wrote into the
  context, plus the `WarningMutation` tuples it applied. It does not advance the
  registry a second time.
- `prepare()` is idempotent per `(peer, episode)`. Calling it twice for the same
  episode returns the cached context result rather than double-counting. The
  registry's `begin_episode` already has this property
  (`mactf/scoring/warning.py`, `WarningRegistry.begin_episode`); the scorer
  enforces it at the context level too, because a caller may reasonably call
  `score_peer` twice for one episode when re-scoring after a ledger edit.

Behaviours:

- Per-rule scoping (W6) is preserved: warnings never pool across rule ids. That
  is `WarningRegistry`'s existing behaviour and this class does not replace it.
- **Per-node scoping** is added: the registry key becomes
  `(node_path, rule_id)` when `node_path` is non-empty. A nano-agent's W4 does
  not decay a root peer's W4. In flat mode `node_path == ""` and the behaviour
  is byte-identical to today. See document 10 section 6.
- Clean streaks: `registry.mark_clean_except(peer, warning_events,
  known_rules=...)` runs once per episode, in `prepare()`, after issuing. The
  `known_rules` set defaults to
  `{rule.id for rule in ALL_RULES if rule.is_penalty}` (`pipeline.py:26`).
- Decay and escalation parameters come from `WarningPolicy`
  (`warning.py:20-29`), never from ad-hoc config, so the W1-W9 contract has one
  owner.
- `state_dict()` serializes every registry keyed by `node_path` then peer, so a
  checkpoint restores warning state exactly (document 09 section 3.3).
- `validate()` rejects `k_esc < 2`, `w_cap` outside `[1, k_esc]`, and
  `n_win`/`n_decay` below 1. `WarningPolicy.__post_init__` already raises for
  these; `validate()` exists so the API can report a clean 422 instead of a
  traceback at scoring time.

### 6.1 Points projection: notional by definition

```python
# mactf/scoring/warning_scorer.py
    def points(self, ctx: ScoreContext, output: ScorerOutput) -> WarningPoints: ...
```

Warnings contribute **nothing** to `final_points`. That is not an oversight and not
a simplification; it is W1, which this document's section 6 already enforces for
`final_score`.

What the method actually computes is `notional`: the number of points the warnings
*would* have cost had they been penalties, at `SCORING.POINTS.WARNING_NOTIONAL`
per warning. It exists for one reason, and the reason is diagnostic:

- A peer with a clean points score and three standing warnings should look
  different from a peer with a clean score and no warnings. The notional figure,
  shown beside `final_points` and never folded into it, is that difference.
- A trainer watching notional rise across epochs sees a cohort heading toward a
  penalty event before the penalty fires. That is the early-warning signal the W
  ladder exists to provide.

`WarningPoints` is `{count, notional, would_be_penalties, deduction: 0.0}`. The
`deduction` field is present and always `0.0` so that a serializer cannot confuse
"no field" with "zero" and a reader cannot wonder whether it was applied.

A test asserts `warning_points.deduction == 0.0` for every fixture, and asserts
`final_points` is bit-identical when `WARNING_NOTIONAL` is set to `0.0`. If that
test ever fails, W1 has been broken.

## 7. `ScoringPipeline`

```python
# mactf/scoring/pipeline.py
class ScoringPipeline(Scorer):
    kind = ScoringKind.PIPELINE
    order = 0

    REQUIRED_KINDS = (ScoringKind.WARNING, ScoringKind.REWARD, ScoringKind.PENALTY)

    def __init__(
        self,
        scorers: Sequence[Scorer] | None = None,
        *,
        score_config: Mapping[str, Any] | None = None,
        severity_weights: Mapping[str, Any] | None = None,
        warning_policy: WarningPolicy | None = None,
        registries: dict[str, WarningRegistry] | None = None,
    ) -> None: ...

    # -- chain management ------------------------------------------------
    @property
    def chain(self) -> tuple[Scorer, ...]: ...
    def add(self, scorer: Scorer) -> "ScoringPipeline": ...
    def remove(self, kind: ScoringKind) -> "ScoringPipeline": ...
    def get(self, kind: ScoringKind) -> Scorer: ...

    # -- scoring ---------------------------------------------------------
    def score_peer(
        self,
        ledger: Ledger,
        *,
        peer: str | None = None,
        episode: int = 0,
        known_rules: set[str] | None = None,
        node_path: str = "",
        depth: int = 0,
        branch_void: frozenset[str] = frozenset(),
        children: tuple[ScorerOutput, ...] = (),
    ) -> PeerScorecard: ...

    def score_episode(
        self,
        ledgers: dict[str, Ledger],
        *,
        episode: int = 0,
        known_rules: set[str] | None = None,
    ) -> EpisodeScorecard: ...
```

`score_peer` is the whole ordering, and it must read in this sequence:

1. Build the base `ScoreContext`.
2. Walk the chain in `order` ascending:
   - call `prepare(ctx)` and keep the returned context,
   - call `score(ctx)` and keep the output,
   - call `finalize(ctx, output)`.
3. Compute `allowed_rules` from `rules.precedence.resolve` and inject it into
   `extras` **before** the penalty stage runs. In practice the pipeline resolves
   precedence once, up front, and each stage reads it. That is why step 2 is
   described as a walk with a pre-computed decision rather than three blind
   calls: the pipeline resolves precedence first, then walks.
4. Compose `ContinuousScore(reward, penalty, warning)` from the three outputs.
   Missing stages raise at construction, not here.
5. Apply reward suppression when the controlling tier admits no reward rule,
   reproducing `continuous.py:217-219`.
6. Call `compliance.constrain(continuous, non_offsettable=...)`, reproducing
   `pipeline.py:103-104`.
7. Merge `branch_void` into the void decision so a descendant breach floors the
   ancestor (section 7.4).
8. Build `PeerScorecard` with the same fields as today
   (`pipeline.py:124-137`), including `suppressed`, `violated_non_offsettable`,
   `severity`, and `metadata["compliance"]`.
9. Emit `signals.episode_peer_scored` with `sender=self`.

With points enabled, three steps are inserted and no existing step changes
number or meaning:

| Inserted after | Step | Failure behaviour |
|---|---|---|
| 1 | `1a.` build `EpisodeEvidence` and `ClaimAudit` from `ctx`, store in `extras` | if the evidence builder raises, points are disabled for this episode with a diagnostic; the docx path proceeds |
| 3 | `3a.` resolve every firing rule's `GuardVerdict` through the guard registry | an unknown rule id is a `KeyError` raised as `ImproperlyConfigured`, never silently unguarded |
| 4 | `4a.` call `points()` on each stage and compose `PointsBreakdown` | a malformed `PointsTable` raises at construction, so this cannot fail per episode |

Step `1a` is before guard resolution because guards consume evidence. Step `4a`
is after `ContinuousScore` composition because points must not be able to affect
it: if `4a` raised before `compliance.constrain`, a points bug would suppress a
scorecard. Ordering it after is what makes "points cannot change `final_score`" a
structural property rather than a promise.

Chain validation, enforced in `__init__` and by `add()`:

| Condition | Result |
|---|---|
| Two scorers with the same `kind` | `ImproperlyConfigured` naming both |
| Two scorers with equal `order` | `ImproperlyConfigured` telling the caller to set an explicit order |
| Any of `REQUIRED_KINDS` missing | `ImproperlyConfigured` naming the missing kind |
| `order < 0` | `ImproperlyConfigured` |
| `prepare()` mutates a context without returning it | Not detectable; the contract is documented and `ScoreContext` is frozen |

`build_scorer` keeps its name and gains an optional chain:

```python
def build_scorer(
    settings: Any,
    *,
    scorers: Sequence[Scorer] | None = None,
) -> ScoringPipeline:
    """Construct a pipeline. With no ``scorers``, the three defaults are used."""
```

`build_scorer(settings)` must return an object whose `score_peer` output is
identical to today's for every input. That is the gate in section 11.

### 7.1 `score_episode_normalized`

```python
    def score_episode_normalized(
        self,
        ledgers: dict[str, Ledger],
        *,
        episode: int = 0,
        known_rules: set[str] | None = None,
        policy: NormalizationPolicy | None = None,
    ) -> tuple[EpisodeScorecard, NormalizationReport]: ...
```

It calls `score_episode`, builds the peer-to-`final_points` map from the returned
scorecard, and passes it through `CohortNormalizer.normalize`. It does not
re-score, and it does not mutate the scorecard it was given: `calibrated` and
`percentile` live on the `NormalizationReport` and in
`PeerScorecard.metadata["normalization"]`, not as new mutable fields on
`PeerScorecard`.

Returning a tuple rather than a fused object is the point. The scorecard is the
per-peer record that gets persisted and compared across resumptions; the report is
a cohort-level derivation that is only meaningful together with its peers. Every
API route that exposes one also exposes the other, so a caller never sees a
`calibrated` value without the stats that produced it.

`score_episode` itself is untouched by normalization. A caller that wants only the
raw rubric calls it and gets no `NormalizationReport`. That is what keeps
`SIGNAL_SOURCE="final_score"` equivalent to `0.1.0`: nothing in the
un-normalized path runs.

### 7.2 `PointsBreakdown` composition and void pinning

Step `4a` is where the three points outputs become one number:

```python
    def score_points(
        self,
        ctx: ScoreContext,
        *,
        reward: RewardPoints,
        penalty: PenaltyPoints,
        warning: WarningPoints,
    ) -> PointsBreakdown: ...
```

```
gross        = reward.total - penalty.total
clamped      = min(MAX_POINTS, max(MIN_POINTS, gross))
final_points = 0.0 if (is_void or violated_non_offsettable) else clamped
```

Three things are load-bearing here.

**The pin is here, not in `compliance.constrain`.** `compliance` decides
`is_void` and `violated_non_offsettable`; it does not know about points, and
teaching it would couple the docx floor arithmetic to the rubric. The pipeline
reads compliance's already-computed verdict and pins. N6 in
[11-points-and-normalization.md](11-points-and-normalization.md) section 7.5 is
the invariant: `final_points` never writes to `is_void`.

**`branch_void` participates.** Step 7 merges descendant voidness into the local
decision before step `4a` reads it, so a hierarchical parent whose child
fabricated is pinned too. Without that ordering a parent could collect points for
an episode whose subtree is void, and document 10's rollup would then average a
pinned zero with an unpinned positive.

**`saturated` is tracked separately.** When `penalty.total > reward.total +
MAX_POINTS`, the clamp hides real magnitude. `PointsBreakdown.saturated` records
it so a report can distinguish "just barely bottomed out" from "capped three
times over". The number a peer sees is still `0.0`; the number an auditor sees is
the raw gross.

`PointsBreakdown` is attached to `PeerScorecard.metadata["points"]` as a plain
dict, so an old serializer that does not know about it still writes a valid
scorecard, and `gross_points` survives even when `final_points` is `0.0` by void
pin rather than by arithmetic. That distinction matters when a peer disputes a
zero: `final_points == 0.0 and gross_points == 62.0` says "you were voided",
while `final_points == 0.0 and gross_points == -14.0` says "you earned that".

### 7.3 Compatibility shim

`mactf/scoring/__init__.py:15` exports `Scorer` from `pipeline.py`. Five call
sites import it that way, including `mactf/orchestration/orchestrator.py:72`
(typed attribute), `mactf/testing/harness.py:91`, and `tests/conftest.py:77`.
The new ABC lives in `base.py` and is also exported as `Scorer`.

`pipeline.py` therefore keeps:

```python
# mactf/scoring/pipeline.py (tail)
from typing import TYPE_CHECKING

if TYPE_CHECKING:                      # static type checkers see the real class
    Scorer = ScoringPipeline           # noqa: F811


def __getattr__(name: str):
    if name == "Scorer":
        warnings.warn(
            "mactf.scoring.pipeline.Scorer now means the abstract base class; "
            "import ScoringPipeline, or from mactf.scoring import Scorer for the ABC",
            DeprecationWarning,
            stacklevel=2,
        )
        return ScoringPipeline
    raise AttributeError(name)
```

Consequences, all intentional:

- `from mactf.scoring.pipeline import Scorer; Scorer(score_config=..., ...)`
  still constructs a working pipeline, with a `DeprecationWarning`.
- `from mactf.scoring import Scorer` gives the ABC. That is the name new code
  should use for `isinstance` and for subclassing.
- `isinstance(obj, ScoringPipeline)` is the correct check for "this is the
  default composite".
- A static checker that follows `TYPE_CHECKING` sees `Scorer` as
  `ScoringPipeline` in `pipeline.py`, which preserves today's annotations in
  `PhaseContext.scorer` and `Orchestrator.scorer`.
- Removal is scheduled for `0.3.0` and recorded in
  [06-verification-and-release.md](06-verification-and-release.md) section 6.

`mactf/scoring/__init__.py` gains exports: `Scorer`, `ScoringKind`,
`ScoreContext`, `ScorerOutput`, `WarningMutation`, `ScorerDiagnostic`,
`RewardScorer`, `PenaltyScorer`, `WarningScorer`, `ScoringPipeline`.

### 7.4 Non-offsettable propagation

The rule, which document 10 section 5 depends on:

> If any node in a branch violated a non-offsettable rule, every ancestor of
> that node is void for the episode and takes the floor. Averaging never
> launders a T0/T1/T2 breach.

Implementation:

- `PenaltyScorer` reports `void_rules` per node.
- `HierarchicalScorer` computes `branch_void(path)` bottom-up: a node's set is
  its own `void_rules` unioned with every child's `branch_void`.
- `ScoringPipeline.score_peer` accepts `branch_void` and, when non-empty,
  passes it to `compliance.constrain` alongside the node's own non-offsettable
  violation, and sets `violated_non_offsettable` on the card to the union.
- In flat mode `branch_void` is always empty and the arithmetic is unchanged.

`PeerScorecard` gains one optional field, `branch_void: tuple[str, ...]`, which
serializes as `branch_void` and defaults to `()`. This is an additive minor
change per [00-scope-and-audit.md](00-scope-and-audit.md) section 6.

## 8. Function wrappers

The free functions stay. `mactf/rl/loss.py:62` (`policy_loss`),
`mactf/rl/loss.py:77` (`compute_loss`), `mactf/rl/advantage.py:30`
(`temporal_difference`), `mactf/rl/advantage.py:41` (`generalized_advantage`),
`mactf/rl/advantage.py:76` (`relative_to_reference`),
`mactf/rl/advantage.py:89` (`penalise_void_advantage`),
`mactf/rl/value.py:44` (`normalise_advantages`), and
`mactf/rl/value.py:61` (`value_loss`) are unchanged, still exported, still
covered by `tests/test_rl.py`. The wrappers call them.

Why wrap at all: a `MACTFTrainer` must accept a user's own objective without
the framework knowing its shape. A `Protocol` plus `from_spec` is what lets the
service layer and the client carry an objective as JSON
(`{"type": "gae", "gamma": 0.99}`) and reconstruct it server-side. Free functions
cannot do that without a dispatch
table scattered across three modules.

### 8.1 `ValueFunction`

```python
# mactf/rl/functions/value.py
class ValueFunction(Protocol):
    def __call__(self, states: torch.Tensor, **kwargs: Any) -> ValueEstimate: ...
    def parameters(self) -> Iterable[nn.Parameter]: ...
    def train(self, mode: bool = True) -> "ValueFunction": ...
    def eval(self) -> "ValueFunction": ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, payload: Mapping[str, Any]) -> None: ...
    def describe(self) -> dict[str, Any]: ...
```

Implementations:

| Class | Behaviour |
|---|---|
| `ModuleValueFunction` | Uses a value head owned by the adapter; matches `PeerAdapter._ensure_value` (`mactf/coopetition/adapter.py:197-223`) including zero-strategy behaviour |
| `CriticValueFunction` | Wraps `ValueHead` (`mactf/rl/value.py:16`) over a hidden vector; a standalone critic with its own optimizer |
| `ReferenceValueFunction` | Frozen `Peer` reference used as a baseline; `parameters()` is empty and `train()` is a no-op |
| `ZeroValueFunction` | Returns zeros with the right shape; valid for pure policy gradient, and it must be explicit rather than an accident |
| `NestedValueFunction` | Composes child value estimates into a parent estimate through a `RollupStrategy`; used by hierarchical training (document 10) |
| `CallableValueFunction` | Adapts `fn(states, **kw) -> Tensor` |

`ValueEstimate` (`mactf/rl/value.py:33`) is reused unchanged. `ValueFunction`
does not add a new estimate type; `mean` and `squeeze()` are already what callers
want.

### 8.2 `AdvantageFunction`

```python
# mactf/rl/functions/advantage.py
class AdvantageFunction(Protocol):
    def __call__(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        *,
        next_values: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        reference: torch.Tensor | None = None,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> AdvantageBatch: ...
    def describe(self) -> dict[str, Any]: ...
```

Implementations and the function each one delegates to:

| Class | Delegates to |
|---|---|
| `TDAdvantage` | `temporal_difference` (`mactf/rl/advantage.py:30`) |
| `GAEAdvantage` | `generalized_advantage` (`mactf/rl/advantage.py:41`) |
| `ReferenceBaselineAdvantage` | `relative_to_reference` (`mactf/rl/advantage.py:76`) |
| `VoidMaskedAdvantage` | `penalise_void_advantage` (`mactf/rl/advantage.py:89`) |
| `ChainAdvantage` | Applies a list in order; validates that each step preserves shape |
| `NormalisedAdvantage` | Post-processes with `normalise_advantages` (`mactf/rl/value.py:44`) |
| `CallableAdvantage` | Adapts a user function |

`ChainAdvantage` is how the current `local_update` behaviour is expressed
without changing it: the existing code computes a per-peer signal, masks it by
void (`phases.py:185-188`), and passes it into the loss. The equivalent chain is
`ReferenceBaselineAdvantage -> VoidMaskedAdvantage`, and
`tests/test_rl.py` must show the same advantages to 1e-12.

Validation, all raising `ValueError` at call time as the free functions already
do: `gamma` and `lam` in `[0, 1]`, matching lengths for rewards and values,
matching shapes for advantages and masks. The wrappers do not relax any check.

### 8.3 `LossFunction`

```python
# mactf/rl/functions/loss.py
class LossFunction(Protocol):
    required_fields: frozenset[str]

    def compute(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, LossBreakdown]: ...
    def describe(self) -> dict[str, Any]: ...
```

`LossBreakdown` (`mactf/rl/loss.py:28`) is reused unchanged; `CompositeLoss`
must produce the same `extras` key `advantage_mean` that
`loss.py:128` produces.

| Class | Behaviour |
|---|---|
| `PolicyGradientLoss` | Clipped surrogate, delegates to `policy_loss` (`loss.py:62`) |
| `ValueRegressionLoss` | `value_loss` (`value.py:61`) with the configured coefficient |
| `KLLoss` | `reverse_kl(policy, reference).mean()`, matching `loss.py:96` |
| `EntropyBonus` | Negative entropy term, matching the `- entropy_coef * ent` sign in `loss.py:103` |
| `ComplianceFloorLoss` | The floor term at `loss.py:106-113`, including `clamp_min(0.0)` |
| `VoidMaskLoss` | Zeroes task credit where `void_mask` is asserted; observability only, as `loss.py:116-119` notes |
| `CompositeLoss` | Weighted sum reproducing `compute_loss` exactly |
| `CausalLMLoss` | `model(**batch).loss` when present, else `F.cross_entropy(logits, labels, ignore_index=-100)`; for Hugging Face fine-tuning |
| `CallableLoss` | Adapts `fn(batch) -> Tensor` or `fn(batch) -> (Tensor, dict)` |
| `ChainedLoss` | Sequential application for multi-objective runs where each term owns its own scaling |

`required_fields` is the guard that makes `MACTFTrainer` fail early:

- `PolicyGradientLoss`: `{"policy", "reference", "actions", "advantages"}`
- `ValueRegressionLoss`: `{"values", "value_targets"}`
- `KLLoss`: `{"policy", "reference"}`
- `EntropyBonus`: `{"policy"}`
- `ComplianceFloorLoss`: `{"compliance_floor"}`
- `CompositeLoss`: the union of its children
- `CausalLMLoss`: `{"input_ids", "labels"}`
- `CallableLoss`: `frozenset(batch_keys)` supplied by the caller

`MACTFTrainer` intersects every configured loss's `required_fields` and checks
the batch *before* the first forward pass. A missing key raises
`ContractError` naming the loss class and the missing key. This is the
difference between a diagnosable failure and a `KeyError` at step 4000.

## 9. Supporting wrappers

These exist so "and so on" in the requirement is a closed list rather than an
open promise. Each is a `Protocol` plus at least one concrete default and each
has `from_spec`.

| Wrapper | Signature | Default implementations |
|---|---|---|
| `ReturnFunction` | `__call__(rewards, *, gamma, masks=None) -> torch.Tensor` | `DiscountedReturn`, `MonteCarloReturn` |
| `RewardFunction` | `__call__(episode, *, peer) -> float` | `ScorecardReward` (reads `PeerScorecard.final_score`), `SignalReward`, `CallableReward` |
| `RegularizationFunction` | `__call__(module) -> torch.Tensor` | `L2Regularization`, `WeightDecayRegularization`, `KLToReferenceRegularization` |
| `ObjectiveFn` | `__call__(batch) -> tuple[Tensor, dict]` | `LossObjective` (adapts a `LossFunction`), `CallableObjective` |
| `GradientFn` | `__call__(parameters, *, max_norm, scaler=None) -> float` | `ClipGradient` (wraps `clip_gradients`, `loss.py:133`), `AccumulateGradient`, `ScaleGradient` |
| `MetricFn` | `__call__(batch, output) -> Mapping[str, float]` | `AccuracyMetric`, `LossMetric`, `PerplexityMetric`, `CallableMetric` |
| `ActionFunction` | `__call__(logits, *, sample) -> torch.Tensor` | `GreedyAction`, `SampledAction` (wrap `mactf/rl/policy.py` helpers) |
| `SynthesisFunction` | `__call__(proposals, *, weights=None) -> Tensor` | `ConsensusSynthesis` (promotes `consensus`, `episode.py:162`), `WeightedConsensus`, `BestByScore`, `LearnedSynthesizer`, `ExplicitSynthesis` |

`GradientFn` deserves a note because it is easy to get wrong with AMP. The
existing AMP path (`phases.py:206-213`) does
`scaler.scale(loss).backward()`, `scaler.unscale_(optimizer)`,
`clip_gradients(...)`, `scaler.step(optimizer)`, `scaler.update()`. That order is
mandatory: clipping after `unscale_` but before `step`. `ClipGradient` encodes
it and `MACTFTrainer` uses `ClipGradient` for both AMP and non-AMP paths. A
custom `GradientFn` that clips before `unscale_` will produce different
gradient norms, and document 08 says so.

## 10. `from_spec` and the spec registry

Every wrapper and scorer implements:

```python
@classmethod
def from_spec(cls, spec: Mapping[str, Any]) -> "Self": ...
```

and every module exposes a registry:

```python
# mactf/scoring/registry.py
SCORER_TYPES: dict[str, type[Scorer]] = {
    "reward": RewardScorer,
    "penalty": PenaltyScorer,
    "warning": WarningScorer,
    # Only valid as the outermost stage of a hierarchical run; the registry
    # accepts it so `ScorerSpec(type="hierarchy")` in the API resolves, and
    # `ScoringPipeline.validate()` rejects it anywhere but the root.
    "hierarchy": HierarchicalScorer,
}

def build_scorer_from_spec(spec: Mapping[str, Any]) -> Scorer: ...
def scorer_capabilities() -> list[dict[str, Any]]: ...
```

```python
# mactf/rl/functions/registry.py
VALUE_TYPES, ADVANTAGE_TYPES, LOSS_TYPES, ROLLUP_TYPES, ...

def build_function(spec: Mapping[str, Any]) -> Any: ...
def function_capabilities() -> dict[str, list[dict[str, Any]]]: ...
```

Spec shape is uniform: `{"type": "<registry key>", ...kwargs}`, where kwargs go
to `__init__`. Unknown keys raise `ValidationError` listing the accepted keys.
An unknown `type` raises `ValidationError` listing the registered keys. There is
no silent fallback to a default type; a typo must never train the wrong
objective.

`scorer_capabilities()` and `function_capabilities()` are what back
`GET /scorers` and `GET /trainers/capabilities`
([02-api-contract.md](02-api-contract.md) section 3.8), so the API cannot
advertise a scorer that does not exist.

Custom Python classes are supported through `{"type": "callable",
"module": "mypkg.scoring", "attr": "MyRewardScorer", "config": {...}}` for
`InProcessTransport` only. `HTTPTransport` rejects it with a 422, because
importing a client-supplied module path on the server is remote code execution.
That asymmetry is documented in [03-python-sdk.md](03-python-sdk.md) section 9
rather than hidden.

## 11. Numerical equivalence obligations

This is the acceptance gate for the whole split.

| Obligation | Oracle | Tolerance |
|---|---|---|
| `RewardScorer` output equals `_reward_breakdown` for every fixture ledger | `tests/test_continuous.py` + `tests/conftest.py` ledgers | exact, then `1e-12` on floats |
| `PenaltyScorer` output equals `_penalty_breakdown`, including Omega | `tests/test_continuous.py` | exact plus `1e-12` |
| `WarningScorer` issues and escalates identically | `tests/test_ledger.py:121-166` | identical rule ids, identical episodes |
| `ScoringPipeline.score_peer` equals today's `Scorer.score_peer` | every ledger in `tests/` | `1e-12` on `final_score`, `floor`; identical `suppressed`, `warnings_issued`, `warnings_escalated` |
| `score_episode` produces identical `EpisodeScorecard` | `tests/test_e2e.py` | `1e-12`, plus identical ordering |
| `CompositeLoss` equals `compute_loss` | `tests/test_rl.py:101-130` | `1e-12` on total and every breakdown field |
| `GAEAdvantage` equals `generalized_advantage` | `tests/test_rl.py:56-70` | `1e-12` |
| `ChainAdvantage(ReferenceBaseline, VoidMasked)` equals the `local_update` signal path | `tests/test_e2e.py` epoch output | `1e-12` |

Two additional gates that are not about floats:

- **Ordering gate.** A test constructs a ledger where a warning matures into a
  penalty in the same episode and asserts: the rule appears in
  `warnings_escalated`, it is *not* in the warning-only exclusion set, and it
  contributes to `P`. Constructing the pipeline with `WarningScorer.order` above
  `PenaltyScorer.order` must raise `ImproperlyConfigured` rather than produce a
  wrong number.
- **No-bypass gate.** A custom `RewardScorer` subclass that reports a large
  positive reward for an episode containing a P1 event must still produce
  `final_score < 0` with `is_void` true. Compliance is not the reward scorer's
  decision.
- **Points non-interference gate.** Running `score_peer` with the points stages
  enabled and with them disabled must produce identical `final_score`, `floor`,
  `is_void`, and `continuous` for every fixture ledger. The strongest form of
  this gate is worth stating, because it constrains the implementation rather
  than the test: `PointsBreakdown` is composed *after* `ContinuousScore` and
  compliance, so the points path has no opportunity to write to them. A test
  that monkeypatches `GuardRegistry.resolve` to raise for every rule must still
  yield byte-identical `final_score` values, and must yield
  `final_points = 0.0` with every rule denied.

### 11.1 Points equivalence obligations

The rubric has its own gates, separate from the numerical ones above:

| Obligation | Oracle | Tolerance |
|---|---|---|
| `RewardPoints.total` matches a hand-computed table for a fixed ledger | a literal expected value in the test, not a recomputation | exact |
| `PenaltyPoints.total` uses superlinear fractions, not `SEVERITY_WEIGHTS` | literal expected values at S1-S4 | exact |
| `PointsBreakdown` for a fixture where nothing fires | `0.0` reward, `0.0` penalty, `0.0` final | exact |
| Void pin | a ledger with a reward and a P1 | `final_points == 0.0` while `final_score` is unchanged from the pre-change oracle |
| `warning_points.deduction` | every fixture | exactly `0.0` |
| denials are exhaustive | every guard that returned `earned=False` appears in `PointsBreakdown.denied` | set equality |
| `score_episode_normalized` returns a scorecard identical to `score_episode`'s | every fixture | `1e-12` on every field except `metadata["normalization"]` |

The last row is the one that keeps normalization from leaking into the record.
If a `calibrated` value ever changes a persisted scorecard field, the tuple return
in section 7.1 has been violated.

## 12. What this document does not change

- `mactf/rules/` - catalog, severity, and precedence are read, never modified.
- `mactf/compliance.py` - the floor arithmetic is untouched. The split calls it
  once, as today.
- `mactf/scoring/continuous.py` - the equation functions stay, and the scorers
  call them. They are not inlined into the scorer classes; that would make the
  numerical-equivalence gate untestable.
- `mactf/scoring/warning.py` - `WarningRegistry` and `WarningPolicy` stay. The
  `WarningScorer` owns *when* they are called, not what they do.
- `mactf/scoring/ledger.py`, `scorecard.py`, `judge.py` - untouched.
- The four `PeerScorecard` fields that the API already serializes. `branch_void`
  is additive and optional.

## 13. Tests

| File | Covers |
|---|---|
| `tests/scoring/test_scorer_base.py` | ABC contract, `ScoreContext.evolve` immutability, `describe`, `state_dict` round-trip |
| `tests/scoring/test_reward_scorer.py` | Bucket arithmetic, severity scaling, ignored buckets, negative-weight rejection |
| `tests/scoring/test_penalty_scorer.py` | Lambda resolution, Omega bounding and overrides, warning-only exclusion, `void_rules` |
| `tests/scoring/test_warning_scorer.py` | W1-W9 transitions, idempotent `prepare`, per-node scoping, decay, escalation, `state_dict` |
| `tests/scoring/test_pipeline_equivalence.py` | The section 11 gate against today's `Scorer` |
| `tests/scoring/test_pipeline_validation.py` | Duplicate kind, duplicate order, missing kind, bad order |
| `tests/scoring/test_shim.py` | `pipeline.Scorer` warns and returns `ScoringPipeline`; `scoring.Scorer` is the ABC |
| `tests/scoring/test_reward_points.py` | Section 4.1 projection, diminishing returns, sorted-order determinism, denial recording |
| `tests/scoring/test_penalty_points.py` | Section 5.1 superlinear fractions, linear accumulation, saturation flag, warning-only exclusion reuse |
| `tests/scoring/test_warning_points.py` | Section 6.1 `deduction == 0.0` under every `WARNING_NOTIONAL`, notional arithmetic |
| `tests/scoring/test_points_composition.py` | Section 7.2 gross/clamp/pin, `gross_points` survives a void pin, `saturated`, `branch_void` participation |
| `tests/scoring/test_points_non_interference.py` | The points non-interference gate in section 11 |
| `tests/rl/test_value_functions.py` | All `ValueFunction` implementations, including `Zero` shape and `Nested` rollup |
| `tests/rl/test_advantage_functions.py` | Delegation to the free functions, chaining, validation errors |
| `tests/rl/test_loss_functions.py` | Each term, `CompositeLoss` equivalence, `required_fields` enforcement |
| `tests/rl/test_function_registry.py` | `from_spec` round-trip, unknown type, unknown key, capabilities output |

## 14. Sequencing

Inside phase A0 of [06-verification-and-release.md](06-verification-and-release.md):

```
base.py -> points.py -> evidence.py -> guards.py
        -> reward_scorer.py -> penalty.py -> warning_scorer.py
        -> pipeline.py (ScoringPipeline + shim) -> equivalence gate green
        -> points composition + non-interference gate green
        -> normalize.py -> score_episode_normalized
        -> __init__.py exports -> hierarchy.py
        -> rl/functions/* -> registries -> capability endpoints unblocked
```

The equivalence gate must be green before `hierarchy.py` starts, otherwise a
hierarchical bug is indistinguishable from a split bug. The points composition
gate must be green before `normalize.py`, because normalization compresses
`final_points` and would happily compress a wrong one.

`points.py`, `evidence.py`, and `guards.py` land before the scorer refactor's
benefit is visible, which is why they are ordered first rather than later: the
`points()` methods on all three scorers need their tables and registries to exist,
and retrofitting them after the equivalence gate is green would reopen a gate that
was already closed. See [11-points-and-normalization.md](11-points-and-normalization.md)
section 13 for the cross-document order.
