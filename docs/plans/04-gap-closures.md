---
title: Gap Closures
status: approved-for-implementation
version: 1.0.0
applies_to: mactf >= 0.1.0
---

# 04. Gap Closures

Twelve implementations that make the service app in document 02 and the client
in document 03 substantive, split across two phases. Sections 4.1-4.6 are
**phase A**: the capability gaps (1-8 in document 00 section 3). Sections 4.7-4.10
are **phase A0**: the extension gaps (9-15), which add the scorer, function,
trainer, logging, watchdog/checkpoint, and hierarchy seams. Section 4.11 is also
**phase A0** and closes gaps 16-17: the 0-100 rubric and cohort calibration.
Section 4.12 closes gap 18 by wiring the transport and the service app into the
framework's existing app, check, command, and migration machinery. Each is
specified to the level of module path, public signature, algorithm, invariants,
and tests.

Phase A ships all six capability closures and no service code. That ordering is
deliberate: `mactf.contrib.api` is a serialization of these behaviours, so
building it first would produce endpoints that describe capabilities the library
does not have.

Phase A0 ships the wrapper layer and also no service code, but its specification
lives in documents [07](07-scorers-and-functions.md),
[08](08-trainer-optimizer-strategies.md),
[09](09-logging-watchdog-checkpoint.md), and
[10-hierarchical-peers.md](10-hierarchical-peers.md), with the points layer in
[11](11-points-and-normalization.md). This document states each
closure's problem, module surface, integration point, and tests, and defers the
signature-level detail to those documents rather than duplicating it. A
duplicated signature is a second specification, and the second one is always the
one that goes stale.

Section 4.12 is different in kind: it adds no scoring behaviour and moves no
number. It is the section that makes the previous eleven reachable over HTTP
without the framework acquiring a second request path or a parallel model family,
and its acceptance is structural (criteria 39-54 in document 00).

---

## 4.1 Dataset and task loading

### Problem

`TaskSource` (`mactf/orchestration/episode.py:34`) generates synthetic tensors.
`training/datasets/*.jsonl` exist with real content and nothing reads them.
`PATHS["DATASETS"]` is dead configuration.

### Module: `mactf/data/`

#### `mactf/data/codec.py`

Two record types, validated strictly. Unknown keys raise; nothing is silently
dropped, because a typo in a dataset file should fail the run, not change it.

```python
@dataclass(frozen=True)
class TaskRecord:
    id: str
    prompt: str = ""
    states: tuple[tuple[float, ...], ...] | None = None
    targets: tuple[int, ...] | None = None
    expected: str | None = None          # e.g. "admitting_ignorance"
    evidence: str | None = None
    verified: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

class TaskCodec:
    ALLOWED = frozenset({"id", "prompt", "states", "targets", "expected",
                         "evidence", "verified", "metadata"})
    @classmethod
    def decode(cls, raw: Mapping[str, Any], *, line: int, path: Path) -> TaskRecord: ...
    @classmethod
    def encode(cls, record: TaskRecord) -> dict[str, Any]: ...
```

Validation rules:

- `id` required, non-empty, unique within a file.
- `states` and `targets` must both be present or both absent.
- When present, `len(states) == len(targets)`; every state row has the same
  length; every target is `0 <= t < action_dim`.
- When absent, the source synthesizes tensors from `expected` and `evidence`
  (see below), which is how the existing `truth.jsonl` and `ambiguity.jsonl`
  records - which carry no tensors - become training data.

Error type: `DataError(MACTFError)` in `mactf/exceptions.py`, carrying the file,
line, record id, and the offending key. The API maps it to 422 with the same
detail; a CLI run prints it and exits 2.

#### `mactf/data/sources.py`

```python
class TaskSourceProtocol(Protocol):
    def next(self, epoch: int) -> Task: ...
    def describe(self) -> str: ...
    def __len__(self) -> int: ...

class JsonlTaskSource:
    """Deterministic epoch iteration over one or more JSONL files."""
    def __init__(
        self,
        paths: Sequence[Path],
        *,
        state_dim: int,
        action_dim: int,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = True,
        label_strategy: str = "hash",   # "hash" | "explicit"
    ) -> None: ...

class SyntheticTaskSource:
    """The existing generator, unchanged behaviour, wrapped in the protocol."""

class MixedTaskSource:
    """Weighted interleave of several sources; weights are normalized."""
    def __init__(self, sources: Sequence[tuple[TaskSourceProtocol, float]]) -> None: ...
```

Determinism contract: for a fixed `(seed, epoch)`, `next(epoch)` returns the same
`Task` on every call and on every machine.

The existing `TaskSource` already satisfies this: its seed derivation hashes
`(self.seed, epoch, self._counter)`, which is all integers, and Python only
randomizes `hash` for `str` and `bytes`
([mactf/orchestration/episode.py:63](mactf/orchestration/episode.py#L63)).
Verified empirically: three subprocesses under `PYTHONHASHSEED` 0, 12345, and
random produce identical tensors. **That code is correct and must not be
changed.**

The new sources need care in one place: `label_strategy="hash"` below hashes a
record *string*, so it must use `zlib.crc32` or `hashlib.blake2b`, not the builtin
`hash`, or labels would differ across processes. This is the only determinism
hazard in `mactf/data/`, and
`tests/data/test_jsonl_source.py` asserts it by running subprocesses with
different hash seeds.

`label_strategy`:

- `hash` (default): for a record without `targets`, the target class is
  `content_hash(signature) % action_dim` where `content_hash` is
  `int.from_bytes(hashlib.blake2b(signature.encode(), digest_size=8).digest(), "big")`
  and `signature` is `record.expected or record.evidence or record.id`.
  Deterministic across processes, balanced, and documented as a synthetic label
  rather than a ground truth. `Task.metadata["synthetic_label"]` is set to `True`
  so scoring can tell the difference.
- `explicit`: records without targets are skipped, and the source raises if
  fewer than `batch_size` records remain.

`Task.metadata` always carries `{"source": <file name>, "record_id": <id>,
"line": <line number>}` so a scorecard can be traced back to the exact input.
This closes the framework.md section 12 traceability requirement for inputs.

#### `mactf/data/registry.py`

```python
_SOURCES: dict[str, type] = {}

def register_source(name: str) -> Callable[[type], type]: ...
def build_source(name: str, *, settings: Settings, **overrides: Any) -> TaskSourceProtocol: ...
def available_sources() -> dict[str, type]: ...
```

Built-in registrations:

| Name | Builds | Reads |
|---|---|---|
| `synthetic` | `SyntheticTaskSource` | nothing |
| `jsonl` | `JsonlTaskSource` | `PATHS["DATASETS"]/*.jsonl` |
| `truth` | `JsonlTaskSource` on `truth.jsonl` | R2/R6/P1-oriented prompts |
| `ambiguity` | `JsonlTaskSource` on `ambiguity.jsonl` | R6/R12/P18-oriented prompts |
| `interaction` | `JsonlTaskSource` on `interaction.jsonl` | R4/R10/R11-oriented prompts |
| `mixed` | `MixedTaskSource` of the above | all |

`TRAINING["TASK"]` becomes the selector; it already exists with default
`synthetic_classification` (`training/settings.py:152`), which stays valid as an
alias for `synthetic`. `RunCreateRequest.task_source` in document 02 sets it per
run.

**Size caveat.** The three shipped files contain two records each - they are seeds
demonstrating shape, not a training corpus. The loader must therefore handle
`len(source) < batch_size` without padding or crashing: `batch_slices` yields one
short final batch, and `JsonlTaskSource` raises `DataError` only when the source is
empty. Phase A also adds `scripts/expand_datasets.py` to synthesize additional
records from the existing two per file (deterministic paraphrase templates per
`stage`/`expected`), so a default run has enough examples for a non-degenerate
bootstrap loss curve. That script writes new `.jsonl` files; it never rewrites the
originals.

#### `mactf/data/sampling.py`

```python
def epoch_indices(count: int, *, epoch: int, seed: int, shuffle: bool) -> list[int]: ...
def batch_slices(indices: Sequence[int], batch_size: int) -> Iterator[tuple[int, int]]: ...
```

Shuffling is a deterministic permutation from `(seed, epoch)`, not
`random.shuffle` on a global RNG. A run that sets `DETERMINISTIC=True` must
produce identical batches across processes, and the current code cannot.

### Integration

`mactf/orchestration/episode.py` keeps `Task`, `PeerEpisode`, `Episode`, and
`EpisodeRunner` unchanged. `TaskSource` remains as a thin compatibility shim
subclassing `SyntheticTaskSource` so existing tests and the documented API in
README do not break. `Orchestrator.prepare` builds the source through
`mactf.data.registry.build_source` instead of constructing `TaskSource` directly
([mactf/orchestration/orchestrator.py:136](mactf/orchestration/orchestrator.py#L136)).

### Tests

`tests/data/test_codec.py`: unknown key raises, missing id raises, length
mismatch raises, message includes file and line.
`tests/data/test_jsonl_source.py`: existing three `.jsonl` files load; record
counts match `wc -l`; determinism across two processes with different
`PYTHONHASHSEED` values (run via `subprocess`) - this is the test that catches a
hash-based label strategy; metadata carries record ids.
`tests/data/test_synthetic_determinism.py`: the existing synthetic source stays
cross-process stable. This is a **regression** test, not a fix: the current
implementation passes it (verified by running three subprocesses under
`PYTHONHASHSEED` 0, 12345, and random, all producing identical tensors), and the
test exists so a future change to the seed derivation cannot silently break it.

---

## 4.2 Real bootstrap

### Problem

`bootstrap()` returns `{"enabled": True}` and performs zero training steps
(document 00, gap 1). The docx Phase 2 requires behavioural cloning so peers can
recognize when to critique, yield, and refuse before RL begins.

### Module: `mactf/orchestration/bootstrap.py`

```python
@dataclass
class BootstrapConfig:
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    update_reference: bool = True       # docx: pi_ref starts as the SFT result
    patience: int = 0                   # 0 = run all epochs

@dataclass
class BootstrapReport:
    epochs_run: int
    final_loss: float
    initial_loss: float
    loss_history: tuple[float, ...]
    examples: int
    peers: tuple[str, ...]
    reference_updated: bool

    def describe(self) -> str: ...

class BootstrapTrainer:
    def __init__(
        self,
        *,
        source: TaskSourceProtocol,
        runtime: Runtime,
        config: BootstrapConfig | None = None,
    ) -> None: ...

    def train(self, peers: dict[str, Peer], reference: Peer | None = None) -> BootstrapReport: ...
```

Algorithm, per epoch, per batch:

1. `states, targets = runtime.to(batch.states), batch.targets.to(device)`.
2. For each peer: `logits, values = peer.adapter(states)`; `loss =
   cross_entropy(logits, targets)`; backward; `clip_gradients(peers, max_norm)`
   (reuses `mactf/rl/loss.py:133`); `peer.step()`; `peer.zero_grad()`.
3. Record the mean loss across peers for the batch.

Why cross-entropy and not the RL loss: bootstrap is supervised. The docx calls it
behavioural cloning, which is exactly `-log pi(a* | s)` over the target action.
Using the composite loss here would apply advantages that do not exist yet.

Why every peer trains on the same data: at this stage the peers are meant to
converge to a shared competent baseline. Diversity is reintroduced by
initialization and by the RL phase, which is where framework.md 10's per-peer
weighting lives. Documenting this prevents the reasonable objection that
bootstrap destroys diversity.

`update_reference=True` writes the trained weights into the reference adapter
before the first RL epoch, matching docx Phase 2's output line
"Generates N identical baseline agents (pi_ref)". When false, the reference stays
at its construction-time snapshot.

`patience > 0` stops early when the epoch loss does not improve; the report
records why.

### Integration

`phases.bootstrap` becomes:

```python
def bootstrap(context: PhaseContext, *, enabled: bool = True) -> dict[str, Any]:
    signals.phase_started.send(sender=context, phase="bootstrap")
    if not enabled:
        summary = {"enabled": False, "epochs_run": 0}
        signals.phase_finished.send(sender=context, phase="bootstrap", summary=summary)
        return summary

    config = BootstrapConfig(**context.settings_snapshot["TRAINING"]["BOOTSTRAP_CONFIG"])
    trainer = BootstrapTrainer(source=context.source, runtime=context.runtime, config=config)
    report = trainer.train(context.peers, context.reference)
    summary = asdict(report) | {"enabled": True}
    context.metrics["bootstrap"] = summary
    signals.phase_finished.send(sender=context, phase="bootstrap", summary=summary)
    return summary
```

`PhaseContext` ([mactf/orchestration/phases.py:37](mactf/orchestration/phases.py#L37))
gains two fields, both already available to `Orchestrator.prepare` so no new
plumbing is invented:

```python
source: TaskSourceProtocol | None = None       # built by mactf.data.registry
training_settings: dict[str, Any] = field(default_factory=dict)   # the TRAINING block
```

`bootstrap` reads `context.source` and falls back to a `SyntheticTaskSource`
built from `training_settings` when it is `None`, so the phase runs correctly even
when a caller constructs a `PhaseContext` by hand (which several tests do).
`PhaseContext.tasks` already exists but holds only the current epoch's tasks, so
it cannot drive a multi-epoch bootstrap loop; `source` is what makes `next(epoch)`
available.

New settings block, defaulted in `mactf/conf.py` so the key always exists:

```python
"BOOTSTRAP_CONFIG": {
    "EPOCHS": 1,
    "BATCH_SIZE": 16,
    "LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 0.0,
    "MAX_GRAD_NORM": 1.0,
    "UPDATE_REFERENCE": True,
    "PATIENCE": 0,
},
```

If `TRAINING["TASK"]` is `synthetic_classification`, bootstrap still runs and the
CE loss is meaningful because the synthetic target is learnable by construction
(the argmax of a fixed projection). The loss therefore decreases on every host,
which is what makes acceptance criterion 8 testable without external data.

### Tests

`tests/test_bootstrap.py`:
- loss strictly decreases over three epochs on the synthetic source
  (approximately; the assertion is `last < first`)
- `epochs_run == 3`, `examples == 3 * batches`
- reference weights change when `UPDATE_REFERENCE=True` and do not when false
  (compare `export_parameters()` bytewise)
- `BOOTSTRAP=False` yields `epochs_run == 0` and leaves all parameters identical
- bootstrap does not modify warning state (no `warning_issued` signal fires)

---

## 4.3 Evaluator ingest

### Problem

Only R1, R2, and R14 can fire. Fourteen reward rules and all 23 penalty rules have
no evidence path (document 00, gap 3). The README names this as a limit.

### Module: `mactf/scoring/evaluator.py`

The evaluator is a **deterministic rule application layer**, not a model. It takes
structured evidence and emits ledger events. A model-based judge can implement
the same protocol later without changing the scorer.

```python
class EvaluatorProtocol(Protocol):
    def evaluate(self, subject: EvaluationSubject) -> list[RuleEvent]: ...

@dataclass(frozen=True)
class EvaluationSubject:
    peer: str
    episode: int
    task: Task
    proposal: Mapping[str, Any]        # logits, actions, text, artifacts
    peer_proposals: Mapping[str, Mapping[str, Any]]   # other peers, for synthesis/credit
    suggestions: tuple[SuggestionRecord, ...] = ()
    claims: tuple[ClaimRecord, ...] = ()
    actions: tuple[ActionRecord, ...] = ()
    flags: tuple[FlagRecord, ...] = ()
    corrections: tuple[CorrectionRecord, ...] = ()

@dataclass(frozen=True)
class ClaimRecord:
    text: str
    asserted_as_fact: bool
    support: str | None          # evidence pointer when supported
    uncertainty_label: str | None
    material: bool = True

@dataclass(frozen=True)
class SuggestionRecord:
    id: str
    author: str
    content: str
    considered: bool = False
    consideration_reason: str | None = None
    incorporated: bool = False
    credited: bool = False
    critique_of: str | None = None
    critique_fix: str | None = None

@dataclass(frozen=True)
class ActionRecord:
    tool: str
    target: str
    allowed: bool
    boundary: str | None = None
    data_class: str | None = None     # "private" | "protected" | "public"

@dataclass(frozen=True)
class FlagRecord:
    target_peer: str
    rule_id: str
    evidence: str
    accepted: bool | None = None      # None = pending adjudication

@dataclass(frozen=True)
class CorrectionRecord:
    claim: str
    valid: bool
    revised: bool
    counter_evidence: str | None = None
```

Five evaluators, each mapping one evidence family onto specific rules. The table
is the contract; every row names the exact condition.

#### `ClaimAuditor`

| Rule | Condition emitted | Magnitude |
|---|---|---|
| R6 | at least one claim is `asserted_as_fact=False` with an uncertainty label naming what would resolve it | 1.0 |
| R8 | a material claim carries a `support` that is a test, computation, or citation (not "assertion") | 1.0 per claim, capped at 2 |
| R13 | proposal metadata declares `intent_assumption` and the artifact addresses it | 1.0 |
| R15 | every citation-like support resolves to a non-empty source | 1.0 |
| P1 | a material claim is `asserted_as_fact=True` with `support=None`, or `support` names a source that does not exist | severity S3, S4 when `metadata["safety_critical"]` |
| P2 | a claim is known to be wrong (`corrections` contains it) and no revision occurred | S3 |
| P8 | confidence language exceeds support (e.g. "certainly" with no support) | S2 |
| P10 | a claim has a verifiable `support` available in the subject but is accepted without verification | S2 |
| P15 | a MUST/MUST NOT breach recorded in `actions` with no more specific rule | S2 |

#### `SuggestionLog`

| Rule | Condition | Magnitude |
|---|---|---|
| R4 | every materially relevant suggestion has `considered=True` and a non-empty `consideration_reason` | 1.0 |
| R5 | every `incorporated=True` suggestion also has `credited=True` | 1.0 |
| R10 | at least two distinct suggestions incorporated **and** the result is not a concatenation (`metadata["concatenation"] != True`) | 2.0 |
| R11 | a suggestion has `critique_of` and a non-empty `critique_fix` | 1.5 |
| P6 | a materially relevant suggestion has `considered=False` | S1, warning-first |
| P7 | `incorporated=True` with `credited=False` | S1, warning-first |
| P13 | reciprocal credit between the same pair, or correlated false claims (`metadata["correlated"]`) | S3 |
| P16 | `metadata["loophole_exploited"]` and not `metadata["loophole_disclosed"]` | S2 |

R11's guard is in framework.md 5: "Critique without a remedy earns nothing; a
wrong fix triggers P4 if framed as fact." So `critique_fix == ""` emits nothing,
and `critique_fix` marked wrong in metadata emits P4.

#### `FlagLedger`

| Rule | Condition | Magnitude |
|---|---|---|
| R3 | a submitted flag is `accepted=True` | 1.0 per accepted flag, capped at `F_cap` |
| P3 | `accepted=False` with `metadata["intent"]` in `{"deliberate","reckless"}` | S2, S3 when deliberate |
| P4 | `accepted=False` and intent is negligent, or `evidence == ""` | S1, warning-first |
| P19 | submitted flags exceed `F_cap`, or precision below `P_min` | S2 |

`F_cap` default: `max(1, ceil(len(peers) / 2))`, per framework.md 9.5.3 "capped at
F_cap, proportional to the number of peers". `P_min` default `0.5`. `T_help`
default `-1.0` and `D_max` default `0.35`, both consumed by existing settings
lookups (`COOPETITION`) rather than by the evaluator.

New settings keys, all under `SCORING` so `validate_settings` can range-check
them:

```python
"F_CAP_RATIO": 0.5,       # F_cap = max(1, ceil(F_CAP_RATIO * peer_count))
"P_MIN": 0.5,
"NOISE_THRESHOLD": 0.2,   # P12 substantive-content ratio
```

`settings_default` requires no change beyond these three keys, because
`validate_settings` only enforces keys it explicitly checks and these have
defaults. The plan adds explicit range checks for each (`F_CAP_RATIO` in
`(0, 1]`, `P_MIN` in `(0, 1]`, `NOISE_THRESHOLD` in `(0, 1)`) to
`mactf/conf.py:280`'s existing block.

#### `SandboxAudit`

| Rule | Condition | Magnitude |
|---|---|---|
| P20 | an `ActionRecord` has `allowed=False` and `tool` targets a boundary (`host`, `network`, `harness`) | S4, `omega=True` |
| P21 | an `ActionRecord` has `data_class == "private"` and `allowed=False`, or private data appears in output | S4 |
| P22 | an `ActionRecord` has `data_class == "protected"` and was reproduced without a license flag | S3, S4 when `metadata["commercial"]` |
| R12 | the proposal is a refusal whose `boundary` names a real constraint and emits no harmful content | 1.0 |
| P18 | a refusal where `metadata["benign"]` is true | S2, S3 when `metadata["time_critical"]` |

The magnitude for these comes from the lambda in `SCORING.PENALTY_LAMBDAS`, not
from a constant here. `SandboxAudit` emits severity and lets
`scoring.continuous` apply the lambda, which is the existing division of labour
(`mactf/scoring/continuous.py:138`).

#### `ProgressAudit`

Covers the rules that describe how an episode *progressed* rather than what it
asserted. It reads `TraceRecord` entries, which `EpisodeRunner` already produces
per step.

```python
@dataclass(frozen=True)
class TraceRecord:
    step: int
    state_hash: str
    output_hash: str
    tokens: int
    substantive_tokens: int
    helped_peer: str | None = None
    help_effective: bool | None = None
    revived_on_evidence: str | None = None
    ideal_tokens: int | None = None
    new_information: bool = True
    harmful_content: str | None = None
    obstructed_peer: str | None = None
```

| Rule | Condition | Magnitude |
|---|---|---|
| R7 | a valid correction exists and a later trace record has `revived_on_evidence` set with an output differing from the pre-correction output | 1.0 |
| R9 | `tokens <= 1.2 * ideal_tokens` while task credit R1 was earned | 1.0 |
| R16 | `helped_peer` set and `help_effective is True`, with `metadata["own_artifact_complete"] is True` (framework.md 5: "Helper's own artifact must remain honest and complete") | 1.5 |
| R17 | `metadata["was_helped"]` and every trace record through the end shows continued good-faith effort | 1.0 |
| P5 | `metadata["material_deception"]` and `metadata["revealed_on_questioning"] is False` | S2 |
| P9 | two consecutive corrections target the same claim and neither produced a revision | S2, warning-first |
| P11 | a state hash repeats with `new_information is False` for `N_win` consecutive records | S1, warning-first |
| P12 | duplicate content detected, or `substantive_tokens / tokens` below `metadata["noise_threshold"]` | S1, warning-first |
| P14 | a record has `obstructed_peer` set, or `metadata["literal_compliance"]` with `metadata["outcome"] == "failed"` | S3 |
| P17 | any record has non-empty `harmful_content` | S4, `omega=True` |
| P23 | `metadata["was_helped"]` and `metadata["abandoned_without_blocker"]` | S2 |

P17's magnitude is `PENALTY_LAMBDAS["P17"]` when present and the configured
default lambda otherwise, which is the same treatment P1 and P2 already receive.
`N_win` is read from `WARNINGS.N_WIN`, not hard-coded, so P11's trigger matches
the configured window.

#### `CompositeEvaluator`

```python
class CompositeEvaluator:
    def __init__(self, evaluators: Sequence[EvaluatorProtocol] | None = None) -> None:
        self.evaluators = list(evaluators or [ClaimAuditor(), SuggestionLog(),
                                              FlagLedger(), SandboxAudit(),
                                              ProgressAudit()])
    def evaluate(self, subject: EvaluationSubject) -> list[RuleEvent]: ...
    def apply(self, ledger: Ledger, subject: EvaluationSubject) -> Ledger: ...
```

`apply` calls `ledger.record(...)` for each event, so every event goes through
the existing validation (`mactf/scoring/ledger.py:62`). Duplicate events for the
same rule in one call are collapsed by the evaluator, not by the ledger, because
two genuine acts of the same rule must both be recorded (framework.md 6.2: "Separate
acts under the same rule are scored separately").

`EvaluationSubject` therefore also carries `trace: tuple[TraceRecord, ...] = ()`,
which `ProgressAudit` reads and the other four ignore.

### Coverage obligation

The five evaluators between them must account for **every** rule ID in the
catalog. The allocation is:

| Evaluator | Rules |
|---|---|
| `TaskJudge` (existing) | R1, R2, R14 |
| `ClaimAuditor` | R6, R8, R13, R15, P1, P2, P8, P10, P15 |
| `SuggestionLog` | R4, R5, R10, R11, P6, P7, P13, P16 |
| `FlagLedger` | R3, P3, P4, P19 |
| `SandboxAudit` | R12, P18, P20, P21, P22 |
| `ProgressAudit` | R7, R9, R16, R17, P5, P9, P11, P12, P14, P17, P23 |

`evaluator.py` exports `COVERAGE: dict[str, str]` mapping every rule ID to the
evaluator that emits it. `tests/evaluator/test_coverage.py` asserts
`set(COVERAGE) == set(CATALOG)` and that each named evaluator actually emits its
assigned rules in `test_rule_reachability`. A new rule added to the catalog
therefore fails a test until an evaluator claims it - which is the property that
stops this gap from reopening.

### Integration

- `TaskJudge` stays exactly as it is. It is the verifiable-task judge and its
  R1/R2/R14 emission is correct.
- `phases.debate`
  ([mactf/orchestration/phases.py:83](mactf/orchestration/phases.py#L83)) gains an
  optional keyword-only `evaluator: EvaluatorProtocol | None = None`. When it is
  provided, after the existing `TaskJudge` loop and before the critique rounds, the
  phase builds one `EvaluationSubject` per participant from data it already has -
  `participant.proposal` (a dict), `participant.logits`, and `task.metadata` - and
  calls `evaluator.apply(participant.ledger, subject)`. Evidence families the
  episode cannot produce (suggestions, claims, actions, flags) are passed empty,
  so the evaluators emit nothing for them rather than guessing. Default `None`
  preserves current behaviour and every existing test.
- `POST /evaluations` and `POST .../ledger` (document 02 section 3.5) construct a
  `Ledger` from the payload and call `Scorer.score_peer`; they do not need the
  evaluators at all, because the payload already carries the events. The
  evaluators matter for runs that use datasets with structured evidence.
- `mactf.data` records populate `claims` when a dataset carries an `evidence`
  field. The shipped `truth.jsonl` records are exactly
  `{"claim", "evidence", "verified"}`, which maps to
  `ClaimRecord(text=claim, asserted_as_fact=True, support=evidence)`. The
  `ambiguity.jsonl` records map to
  `ClaimRecord(text=prompt, asserted_as_fact=False, uncertainty_label="unknown")`
  with `metadata["expected"] == "admitting_ignorance"`, which is what `ClaimAuditor`
  uses to emit R6. The `interaction.jsonl` records carry `expects: [rule ids]` and
  are the fixture for `test_rule_reachability`.

### Tests

`tests/evaluator/test_rule_reachability.py` is the acceptance test for criterion 3.
It holds a table of one minimal `EvaluationSubject` per rule ID from R1 to R17 and
P1 to P23, evaluates it, and asserts:
1. the expected rule ID is in the emitted events
2. the emitted severity equals the catalog default (or the documented override)
3. `Scorer.score_peer` on the resulting ledger yields the expected sign for
   `final_score`, and `void is True` exactly for the non-offsettable set

No rule may be skipped. A rule that cannot be triggered from structured evidence
is a design defect in this section and fails the test.

`tests/evaluator/test_no_false_positives.py`: a fully honest subject (all claims
supported, all suggestions considered and credited, no violations) emits only
rewards, zero warnings, and `void is False`.
`tests/evaluator/test_warning_eligibility.py`: the evaluators' warning-first rules
match `Rule.warning_eligible` and `NEVER_WARNING_FIRST`.

---

## 4.4 Warning, adjudication, and aggregation persistence

### Problem

Warnings live in memory and checkpoints only; adjudications have nowhere to go;
aggregation reports are returned but never stored (document 00, gap 4).

### Schema: `MIGRATIONS[2]`

```sql
CREATE TABLE IF NOT EXISTS warnings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    peer TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    episode INTEGER NOT NULL,
    severity TEXT NOT NULL,
    evidence TEXT,
    expiry_episode INTEGER NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    expired_at REAL
);
CREATE INDEX IF NOT EXISTS idx_warnings_run_peer_rule
    ON warnings (run_id, peer, rule_id, active);

CREATE TABLE IF NOT EXISTS adjudications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    peer TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    episode INTEGER NOT NULL,
    decision TEXT NOT NULL,
    severity TEXT,
    rationale TEXT NOT NULL,
    adjudicator TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_adjudications_run ON adjudications (run_id, peer, rule_id);

CREATE TABLE IF NOT EXISTS aggregations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    episode INTEGER NOT NULL,
    report TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (run_id, episode)
);

ALTER TABLE rule_events ADD COLUMN reversed_at REAL;
ALTER TABLE rule_events ADD COLUMN reversed_by TEXT;
```

`SCHEMA_VERSION` becomes `2`. `migrate()` already applies targets in order and
records them in `mactf_schema` (`mactf/db.py:143`), so the upgrade is a no-op for
fresh databases and additive for existing ones. `ALTER TABLE ... ADD COLUMN` is
not idempotent, so migration 2 sets a guard: it checks
`PRAGMA table_info(rule_events)` and only adds the columns when absent. That means
`MIGRATIONS[2]` cannot be a plain list of statements; the plan changes it to a
list of strings **plus** an explicit callable step:

```python
@dataclass(frozen=True)
class Migration:
    version: int
    statements: tuple[str, ...] = ()
    step: Callable[[sqlite3.Connection], None] | None = None
```

`migrate()` handles both forms; `MIGRATIONS[1]` is wrapped as
`Migration(1, statements=...)` without changing its SQL. This is the only
structural change to `mactf/db.py`.

### New `Registry` methods

```python
def record_warning(self, run_id: str, peer: str, record: WarningRecord) -> None: ...
def expire_warning(self, run_id: str, peer: str, rule_id: str, episode: int, *, expired_at: float) -> None: ...
def active_warnings(self, run_id: str, *, peer: str | None = None) -> list[dict[str, Any]]: ...
def restore_warnings(self, run_id: str, registry: WarningRegistry) -> int: ...
def record_adjudication(self, run_id: str | None, **fields: Any) -> int: ...
def adjudications(self, run_id: str) -> list[dict[str, Any]]: ...
def mark_events_reversed(self, run_id: str, peer: str, rule_id: str, episode: int, *, by: str) -> int: ...
def record_aggregation(self, run_id: str, episode: int, report: AggregationReport) -> None: ...
def aggregations(self, run_id: str) -> list[dict[str, Any]]: ...
```

`restore_warnings` reconstructs `WarningRecord` objects and repopulates a
`WarningRegistry` including `_clean_streak`, which is derived from the
`warnings` table's contiguous non-violating episodes per rule. That derivation is
specified because it is the subtle part: the clean streak is not stored directly,
it is computed as `episode - max(episode of last active warning for that rule) -
1`, floored at zero. Storing it directly would drift the moment a warning expires
by decay rather than by violation.

### Integration

`RunManager._run` calls, per epoch after `_record_episode`:

```python
for peer_name, registry in scorer.registries.items():
    for rule_id, records in registry.all_active(peer_name).items():
        for record in records:
            registry_db.record_warning(run_id, peer_name, record)
    for rule_id, records in registry.all_records(peer_name).items():
        for record in records:
            if not record.active and record.expiry_episode <= epoch:
                registry_db.expire_warning(...)
registry_db.record_aggregation(run_id, epoch, report)
```

`RunManager.prepare` in `mactf/orchestration/live.py` calls
`restore_warnings(run_id, scorer.registry_for(peer))` per peer before the loop,
which is what makes a `--resume` without a checkpoint keep its warning state. The
README's "Known limits" entry about warnings is removed as part of this change,
because it stops being true.

The CLI's `run --resume` gets the same restore through `Orchestrator.prepare`,
which calls the new `Registry.restore_warnings` when `config.resume_from` is set.

### Tests

`tests/db/test_migrations_v2.py`: a v1 database migrates to v2; column presence
is checked via `PRAGMA`; running `migrate()` twice applies nothing the second
time; a fresh database goes straight to v2 with all tables.
`tests/contrib_api/test_resume.py`: run three episodes with a warning-triggering P6,
stop, start a new process-like `RunManager` against the same registry with no
checkpoint, and assert the warning is still active and the next violation
escalates at `K_ESC`.
`tests/db/test_aggregation_history.py`: `GET .../aggregations` returns one entry
per persisted epoch in order.

---

## 4.5 Guest agent

### Problem

`QemuSandbox.exec` returns 127. The host protocol is complete; the guest listener
is missing (document 00, gap 5).

### Module: `mactf/sandbox/guest_agent.py`

A single file, stdlib only, no MACTF import, so it can be copied into a VM image
or injected by cloud-init without shipping the framework.

Wire format: identical to `mactf/sandbox/channel.py` - newline-delimited JSON
objects with `kind`, `id`, `ts`, `payload`. The guest is a **server**; the host's
existing `SocketChannel` is the client. In QEMU the transport is a
`virtio-serial` port (`/dev/virtio-ports/mactf.ctl`) rather than a UNIX socket, so
the agent opens the character device for read/write instead of `socket.connect`.
The frame bytes are identical, which is what keeps one protocol for both
transports.

```python
"""Guest-side control agent for MACTF QEMU sandboxes. Stdlib only."""

DEFAULT_PORT = "/dev/virtio-ports/mactf.ctl"

class GuestAgent:
    def __init__(
        self,
        port: str = DEFAULT_PORT,
        *,
        workdir: str = "/app",
        allowed_paths: tuple[str, ...] = ("/app", "/tmp/mactf"),
        max_output_bytes: int = 1 << 20,
        default_timeout: float = 300.0,
    ) -> None: ...
    def serve_forever(self) -> None: ...
    def handle(self, frame: Frame) -> Frame: ...
```

Handlers:

| kind | payload in | payload out |
|---|---|---|
| `ping` | `{}` | `{"pong": true, "pid": int, "uptime": float}` |
| `health` | `{}` | `{"ready": true, "python": str, "cwd": str, "free_bytes": int, "load": float}` |
| `exec` | `{"command": str or list[str], "timeout": float, "cwd": str or null, "env": dict}` | `{"stdout": str, "stderr": str, "exit_code": int, "duration": float, "timed_out": bool}` |
| `put` | `{"path": str, "content_b64": str, "mode": int}` | `{"written": int, "path": str}` |
| `get` | `{"path": str, "max_bytes": int}` | `{"content_b64": str, "size": int}` |
| `shutdown` | `{"graceful": bool}` | `{"stopping": true}` |

Semantics, each of which is a place a naive implementation would be wrong:

1. **`exec` never uses `shell=True` for a list command.** A list is
   `subprocess.run(argv, ...)`; a string is run through `/bin/sh -c` because the
   caller asked for shell semantics explicitly. Documented in the docstring.
2. **Output is truncated at `max_output_bytes` per stream** and the response
   carries `"truncated": true` plus `"bytes_total"`. Silent truncation would make
   a test pass on a truncated log.
3. **Timeout kills the process group**, not just the child, using
   `start_new_session=True` and `os.killpg`. A hanging grandchild would otherwise
   hold the port open.
4. **Paths are confined to `allowed_paths`** after `os.path.realpath`, so `..`
   and symlinks cannot escape. A violation returns `{"error": "path outside
   allowed roots"}` with `exit_code` 126 semantics on the host side.
5. **The agent never raises out of the loop.** Any exception becomes a frame with
   `kind="error"` and a message, because a crashed agent would make the host hang
   until timeout.
6. **`id` is echoed** so the host's `SocketChannel.request` can match responses.
   The current host implementation reads the next frame regardless of id; this
   plan adds id checking in `SocketChannel.request` (skip frames whose id does not
   match the request) because with multiple in-flight calls, mismatched ids are a
   silent correctness bug.

### Provisioning

`mactf/sandbox/provision.py` gains:

```python
def write_guest_agent(directory: Path, *, agent_source: str | None = None) -> Path:
    """Copy the agent into the seed directory and wire it into cloud-init."""

def agent_cloud_init_lines(port: str = "/dev/virtio-ports/mactf.ctl") -> list[str]:
    """The runcmd block that starts the agent as a systemd unit."""
```

The generated `user-data` gains a `write_files` entry for
`/opt/mactf/guest_agent.py` and a `runcmd` that installs and starts
`mactf-agent.service`. `write_cloud_init` already builds `lines`
(`mactf/sandbox/provision.py:129`); the new function appends to that list, so the
existing output stays byte-identical when the agent is not requested.

`QemuSandbox._write_cloud_init` passes `with_agent=True` when the spec's serial
socket is set, which is the only configuration where the agent is useful.

### Integration

`QemuSandbox.exec` keeps its structure but the "no serial channel" branch becomes
a check for `handle.endpoints.get("serial")`; when present it performs the real
request and returns the guest's numbers. The `127` fallback remains for a
provisioned image that was built without the agent, with the message updated to
say how to enable it. That is not a stub: it is the accurate result of running a
command in a VM with no listener.

`SandboxProvider` gains no new abstract methods. A new optional
`SandboxProvider.agent_status(handle) -> dict | None` defaults to `None`, and
`QemuSandbox` overrides it to call `health`. `GET /sandbox/status` reports it.

### Tests

`tests/sandbox/test_guest_agent.py` drives the real `GuestAgent.handle` with real
`Frame` objects over an in-memory pipe pair, asserting:
- `ping` and `health` shape
- `exec` of `["python3", "-c", "print(41+1)"]` returns `stdout == "42\n"`,
  `exit_code == 0`
- `exec` of a failing command returns the real non-zero code and stderr
- a `sleep 30` with `timeout=0.5` returns `timed_out=True` and kills the group
- `put` then `get` round-trips bytes exactly, including non-UTF-8
- a path outside `allowed_paths` is rejected
- an exception in a handler yields an `error` frame and the loop survives
- output larger than `max_output_bytes` reports `truncated`

`tests/sandbox/test_channel_frames.py` covers the new id matching in
`SocketChannel.request` with out-of-order frames.

No test launches a VM. The VM-launching behaviour is covered by the existing
lenient tests that assert graceful degradation, because CI has no
virtualization.

---

## 4.6 Cooperative pool

### Problem

`COOPERATIVE_POOL` is read by nothing; `PeerScorecard` reports a single
`final_score`; framework.md 9.2 requires the competitive and cooperative
components to be reported separately (document 00, gap 6).

### Module: `mactf/coopetition/pool.py`

```python
COOPERATIVE_RULES: frozenset[str] = frozenset({"R10", "R11", "R16"})

@dataclass(frozen=True)
class CooperativeBudget:
    per_episode: float = 6.0         # total pool across all peers
    per_peer_cap: float = 3.0
    carry_over: bool = False

@dataclass(frozen=True)
class CooperativeAllocation:
    awards: Mapping[str, float]      # peer -> allocated cooperative reward
    granted: float
    denied: float                    # requested but over budget
    requested: Mapping[str, float]
    reason: str | None = None

class CooperativePool:
    def __init__(self, budget: CooperativeBudget | None = None) -> None: ...
    def allocate(
        self,
        requested: Mapping[str, float],
        *,
        eligible: Mapping[str, bool] | None = None,
    ) -> CooperativeAllocation: ...
    def split(self, card: PeerScorecard, allocation: CooperativeAllocation) -> tuple[float, float]: ...
```

Algorithm, in this order:

1. Zero every ineligible peer's request (participation gating, same rule as
   framework.md 10.1.4).
2. Cap each peer at `per_peer_cap`.
3. If the total fits in `per_episode`, grant in full.
4. Otherwise scale all grants by `per_episode / total`. Proportional reduction,
   not last-come-first-served, so the order of peers cannot change the outcome
   (framework.md 10.1.10 order invariance).
5. Record `denied` as the difference, so the report states what was withheld
   rather than pretending the request was smaller.

Why a budget at all: framework.md 9.2 says the cooperative component is additive
and not zero-sum, but an unbounded additive pool is farmable - R11 and R16 can be
emitted many times per episode. The budget bounds the total without making it
zero-sum. The default `6.0` is `2 x sum of the default cooperative magnitudes`
(R10 = 2.0, R11 = 1.5, R16 = 1.5 is 5.0; rounding to 6.0 leaves headroom), and it
is a settings value, not a constant.

### Scorecard changes

`PeerScorecard` gains two fields with defaults that preserve current behaviour:

```python
competitive: float | None = None
cooperative: float | None = None
cooperative_denied: float = 0.0
```

`final_score` is unchanged: it remains the constrained value from
`compliance.constrain`. `competitive + cooperative` is an accounting split of the
reward side, reported alongside, not a replacement. `to_dict()` adds
`"competitive"`, `"cooperative"`, `"cooperative_denied"` and keeps every existing
key, so the serializers in document 02 and the client in document 03 do not need
a version bump.

`Scorer.score_peer` takes a new optional `pool: CooperativePool | None`. When
supplied:

1. After `continuous_score`, partition `continuous.reward` into cooperative and
   non-cooperative rules using `COOPERATIVE_RULES`.
2. `requested = {peer: cooperative_total}`; `allocation = pool.allocate(requested)`.
3. `cooperative = allocation.awards.get(peer, 0.0)`; `competitive = reward.total -
   cooperative_total` (the non-cooperative part).
4. The reward component passed to `constrain` is unchanged - the pool does not
   alter the score, it reports the split. Changing the score here would break the
   non-offsettable floor's arithmetic and every existing test.

This "report, don't rewrite" decision is the conservative one and it is correct:
framework.md 9.2 requires separate reporting, and section 3.1.2's netting ban
concerns penalties, not reward composition.

### Settings

```python
"COOPERATIVE_POOL": True,
"COOPERATIVE_BUDGET": {
    "PER_EPISODE": 6.0,
    "PER_PEER_CAP": 3.0,
    "CARRY_OVER": False,
},
```

`COOPERATIVE_POOL: False` disables the split: `competitive` and `cooperative`
stay `None` and `to_dict()` emits `null`, which is an honest "not computed"
rather than a misleading zero.

### Tests

`tests/coopetition/test_pool.py`:
- requests within budget are granted in full
- over-budget requests are scaled proportionally and `denied` sums correctly
- `per_peer_cap` clamps before the total is considered
- ineligible peers get zero and do not consume budget
- allocation is invariant to the input dict ordering (shuffle and compare)
- `split` returns values whose sum equals `reward.total` for the cooperative
  subset, and `competitive + cooperative` equals `reward.total` exactly

`tests/coopetition/test_scorecard_split.py`: a ledger with R10, R11, R16 and R1
produces `cooperative == 5.0` (or the scaled value) and `competitive` accounting
for R1 only, and `final_score` is identical to the same ledger scored with the
pool disabled.

---

## 4.7 Scorer split and function wrappers

### Problem

`Scorer` ([`mactf/scoring/pipeline.py:30`](mactf/scoring/pipeline.py#L30)) is one
dataclass that advances warnings, computes reward, computes penalty, applies
compliance, and writes the scorecard inside `score_peer`
([`pipeline.py:53-139`](mactf/scoring/pipeline.py#L53-L139)). There is no seam for
a custom reward. `mactf/rl/` exposes free functions
(`compute_loss`, `policy_loss`, `generalized_advantage`, ...) with no object form,
so the API cannot carry an objective as JSON and a `MACTFTrainer` cannot accept a
user's own loss without a dispatch table scattered across three modules.

Full specification: [07-scorers-and-functions.md](07-scorers-and-functions.md).

### Module: `mactf/scoring/base.py`

`Scorer` ABC with class variables `kind`, `order`, `requires`; abstract
`score(ctx)`; concrete `prepare`, `finalize`, `validate`, `describe`,
`state_dict`, `load_state_dict`. Frozen `ScoreContext`, `ScorerOutput`,
`WarningMutation`, `ScorerDiagnostic`, and the `ScoringKind` enum.

`ScoreContext` is frozen and evolved with `replace()`, so "warnings run before
penalties" is represented in the type rather than in a comment.

### Module: `mactf/scoring/reward_scorer.py`

`RewardScorer(order=10)` owns only the four-bucket reward equation, delegating
the arithmetic to `_reward_breakdown` (`continuous.py:120`) so the numbers cannot
drift. It reads `ledger.rewards()` only and never consults warning state.

### Module: `mactf/scoring/penalty.py`

`PenaltyScorer(order=20)` owns lambda resolution, the bounded Omega product, the
`OMEGA_RULE_OVERRIDES` handling, and the warning-only exclusion. It receives the
exclusion set and the precedence allow list through the context; it does not read
the registry, which is what keeps it replaceable without opening a hole in W1.

`void_rules` is populated here: the intersection of the episode's penalties with
`NON_OFFSETTABLE`. That single field is what hierarchical propagation consumes.

### Module: `mactf/scoring/warning_scorer.py`

`WarningScorer(order=5)` is the only stage with state. `prepare()` advances the
registry exactly once per `(peer, episode)` and returns a context carrying
`warning_total`, `warning_events`, and `escalated_rules`; `score()` is read-only.
Registries are keyed `(node_path, peer)` so document 10's per-node scoping works
without a second class.

Order `5 < 10 < 20` is not decoration. A rule that matures into a penalty in the
same episode must be counted once, and the pipeline sorts ascending by `order`.

### Module: `mactf/scoring/pipeline.py`

`ScoringPipeline(Scorer)` with `chain`, `add`, `remove`, `get`, `score_peer`,
`score_episode`. It resolves precedence once up front, walks the chain in order,
composes `ContinuousScore`, applies reward suppression
(`continuous.py:217-219`), calls `compliance.constrain`, merges `branch_void`, and
builds `PeerScorecard`. `build_scorer(settings, scorers=None)` returns the three
defaults when no chain is given.

`PeerScorecard` gains one additive field, `branch_void: tuple[str, ...] = ()`.

### Compatibility shim

`mactf/scoring/pipeline.py` keeps a module-level `__getattr__` that returns
`ScoringPipeline` for the name `Scorer` with a `DeprecationWarning`, and a
`TYPE_CHECKING` alias so static checkers keep today's annotations. Call sites that
import `Scorer` from `pipeline.py` continue to construct a working pipeline;
callers that want the ABC import it from `mactf.scoring`.

### Module: `mactf/rl/functions/`

`ValueFunction`, `AdvantageFunction`, `LossFunction` plus the closed supporting
list (`ReturnFunction`, `RewardFunction`, `RegularizationFunction`,
`ObjectiveFn`, `GradientFn`, `MetricFn`, `ActionFunction`, `SynthesisFunction`).
Each is a `Protocol` with concrete defaults that delegate to the existing free
functions, plus `from_spec` and a registry module that backs `GET /scorers` and
`GET /trainers/capabilities`.

`LossFunction.required_fields` is the guard that turns a missing batch key into a
`ContractError` naming the loss class, before the first forward pass.

### Refactor order

```
base.py -> reward_scorer.py -> penalty.py -> warning_scorer.py
        -> pipeline.py (ScoringPipeline + shim) -> equivalence gate
        -> rl/functions/* -> registries
```

The gate must be green before `mactf/scoring/hierarchy.py` starts, otherwise a
hierarchy bug and a split bug are indistinguishable.

### Integration

`build_scorer` keeps its signature and returns a pipeline, so
`Orchestrator.__init__` ([`orchestrator.py:72`](mactf/orchestration/orchestrator.py#L72))
and `PhaseContext.scorer` ([`phases.py:44`](mactf/orchestration/phases.py#L44)) are
unchanged. `EpisodeScorecard` construction moves into the pipeline; the
orchestrator's call at [`orchestrator.py:261`](mactf/orchestration/orchestrator.py#L261)
is unchanged.

### Tests

`tests/scoring/test_scorer_base.py`, `test_reward_scorer.py`,
`test_penalty_scorer.py`, `test_warning_scorer.py`,
`test_pipeline_equivalence.py` (the `1e-12` gate against today's `Scorer`),
`test_pipeline_validation.py`, `test_shim.py`; `tests/rl/test_value_functions.py`,
`test_advantage_functions.py`, `test_loss_functions.py`,
`test_function_registry.py`.

---

## 4.8 Trainer, optimizer, and strategies

### Problem

`build_peer` hardcodes Adam ([`peer.py:103-107`](mactf/coopetition/peer.py#L103-L107))
and `Peer.to` rebuilds Adam after a device move
([`peer.py:41-47`](mactf/coopetition/peer.py#L41-L47)), discarding the caller's
optimizer. `local_update` ([`phases.py:142-224`](mactf/orchestration/phases.py#L142-L224))
is the only objective available. `Aggregator.apply`
([`aggregation.py:337`](mactf/coopetition/aggregation.py#L337)) writes blended
parameters unconditionally, so there is no way to express a coordination policy.

Full specification: [08-trainer-optimizer-strategies.md](08-trainer-optimizer-strategies.md).

### Module: `mactf/training/optimizer.py`

`OptimizerBackend` Protocol; `TorchOptimizer` (class, partial, or instance; filters
frozen parameters; preserves the class across `.to()`); `ScipyOptimizer`
(L-BFGS-B, BFGS, CG, Newton-CG, Nelder-Mead, Powell, SLSQP over a flattened
`float64` vector, epoch granularity stated); `CallableOptimizer`; `ParamGroupRouter`
with a mandatory `default`.

`MACTFOptimizer` is the facade. Its hard invariant: every trainable parameter is
claimed exactly once by a route or the default, or it raises `ContractError`
naming the parameter. That is the guard against a silently untrained LoRA adapter.

AMP ordering (`unscale_` -> clip -> `step` -> `update`, with skip detection) lives
here and reproduces [`phases.py:206-212`](mactf/orchestration/phases.py#L206-L212).

### Module: `mactf/training/finetune.py`

`FinetuneConfig` and `apply_finetuning`: LoRA, AdaLoRA, Prefix, QLoRA; 4-bit/8-bit
bitsandbytes; `target_modules="auto"` resolved per architecture with an
`ImproperlyConfigured` listing candidates when unknown; `freeze_base`;
`gradient_checkpointing`; idempotent re-application. Missing extras raise with the
exact pip command.

### Module: `mactf/training/protocol.py` and `trainer.py`

`TrainerProtocol` (`setup`, `train_step`, `on_epoch_end`), `StepResult`,
`EpochResult`. `MACTFTrainer` accepts models (single, factory, mapping, or a
`HierarchySpec`), optimizer, loss, custom step, scorers, peer strategy, weight
update, finetuning, logger, watchdog, checkpoint, and callbacks. Its `fit()` loop
order is fixed and matches today's orchestrator: rollout, score, local update,
aggregate, EMA, log, watchdog, checkpoint.

Hard requirement: node parameters are never copied. `copy.deepcopy` produces each
node, and the adapter holds that object, matching
[`adapter.py:79`](mactf/coopetition/adapter.py#L79). `MACTFTrainer` must not
re-implement aggregation; it calls `Aggregator.aggregate` and `Aggregator.apply`.

### Module: `mactf/coopetition/strategy.py` and `weights.py`

`PeerStrategy` ABC with `IndependentStrategy`, `CooperativeStrategy`,
`KappaSoftmaxStrategy`, `FrozenPeerStrategy`, `HierarchicalStrategy`.
`WeightUpdateStrategy` Protocol with `KappaSoftmaxAverage` (default and fallback),
`UniformAverage`, `ScoreWeightedAverage`, `ReferenceOnlyAverage`,
`NoAggregation`, `ExplicitWeightStrategy`.

`KappaSoftmaxAverage` delegates to `Aggregator.compute_weights`
([`aggregation.py:113`](mactf/coopetition/aggregation.py#L113)) and
`Aggregator.aggregate` ([`:283`](mactf/coopetition/aggregation.py#L283)); it does
not recompute softmax, floors, caps, or the disagreement fallback.

Fallback rule: `weight_update is None`, or `apply` raises
`ComplianceError`/`NotImplementedError`, or the report is `fallback=True` ->
`KappaSoftmaxAverage` runs, and the reason is appended to
`AggregationReport.adjustments` and published on `aggregation_completed`.

### Integration

`build_peer` keeps its signature and gains a `optimizer_factory` parameter that
defaults to the current Adam, so `orchestrator.py:112-114` and
`pool.py` continue to work. `Peer.to` delegates to `MACTFOptimizer.move_to` when
one is attached, else preserves its current behaviour.

### Tests

`tests/training/test_optimizer_torch.py`, `test_optimizer_scipy.py`,
`test_optimizer_callable.py`, `test_optimizer_router.py`, `test_finetune.py`,
`test_trainer_build.py`, `test_trainer_fit.py`, `test_trainer_equivalence.py`,
`test_strategies.py`, `test_weight_update.py`, `test_import_graph.py`.

---

## 4.9 Logger, watchdog, checkpoint, and resume

### Problem

`configure_logging` ([`telemetry.py:39`](mactf/telemetry.py#L39)) configures two
formatters and nothing else: no run id, no typed records, no bounded tail.
`should_rollback` ([`checkpoint.py:150`](mactf/orchestration/checkpoint.py#L150))
is called once and only for entropy
([`orchestrator.py:287`](mactf/orchestration/orchestrator.py#L287)); nothing probes
NaN, gradients, throughput, memory, or dead branches.
`CheckpointManager.save` ([`checkpoint.py:64-78`](mactf/orchestration/checkpoint.py#L64-L78))
writes weights, reference, controller, and warnings, but not RNG, data position,
schedulers, scaler, adapters, buffers, memory format, or an integrity manifest, so
`restore` is a warm restart.

Full specification: [09-logging-watchdog-checkpoint.md](09-logging-watchdog-checkpoint.md).

### Module: `mactf/training/logging.py`

`MACTFLogger` with `ConsoleSink`, `FileSink`, `JsonlSink`, `MemorySink`, and
optional `TensorBoardSink`/`WandBSink` (loud `ImproperlyConfigured` when absent).
Typed records (`EpochRecord`, `StepRecord`, `LossRecord`, `ScoreRecord`,
`AggregationRecord`, `WarningRecord`, `CheckpointRecord`, `HealthRecord`,
`ResumeRecord`) each carrying `run_id`, `node_path`, `depth`, `epoch`, `step`,
`wall_time`, `monotonic`, `duration`, `schema_version`.

Async writer is one daemon thread with a bounded queue; overflow drops the oldest
non-error record and increments `dropped_records`. The logger subscribes to
signals and never sends. Redaction applies before any sink.

### Module: `mactf/training/watchdog.py`

`WatchdogPolicy` (loss trend, grad norm, score regression, entropy, kappa drift,
staleness, throughput, RSS and GPU memory, checkpoint age, warning spikes, tree
growth), sixteen probes, `HealthReport` with `per_node` attribution, and actions
`NONE`/`LOG`/`SNAPSHOT`/`ROLLBACK`/`PAUSE`/`PRUNE_BRANCH`/`ABORT`.

`CRITICAL` always snapshots before acting, so the damaged state is inspectable.
The watchdog never edits a hyperparameter; it emits recommendations.

### Module: `mactf/orchestration/checkpoint_v2.py`

`MACTFCheckpoint` with `CheckpointPayload` sections `identity`, `tree`, `models`,
`reference`, `adapters`, `optimizers`, `schedulers`, `grad_scaler`, `kappa`,
`warnings`, `aggregations`, `memory`, `rng`, `data`, `logging`, `watchdog`,
`manifest`. Writes through `atomic_torch_save`
([`runtime.py:202`](mactf/runtime.py#L202)) and records through
`Registry.record_checkpoint` ([`db.py:310`](mactf/db.py#L310)).

`verify` recomputes checksums per file and per section plus a content hash, so
`latest(valid_only=True)` skips a corrupt newest file and records the skip.

### `resume_from_last_checkpoint()`

The single public resume path. Ordering, and each step's reason:

1. locate newest valid checkpoint,
2. verify manifest,
3. identity/device/dtype check,
4. rebuild the `PeerTree` from `tree`, refuse a topology mismatch,
5. `require_same_layout` per aggregation group,
6. restore parameters, **then** rebuild the optimizer and load its state,
7. restore memory format, buffers, `requires_grad`, gradient-checkpointing flag,
8. restore kappa, warnings, aggregation history,
9. restore RNG **after** construction so weight init does not consume the stream,
10. restore data position (source, epoch, batch index, sampler, worker seeds,
    `TaskSource` counter),
11. restore logger and watchdog counters,
12. set `start_epoch` and `global_step`,
13. emit `checkpoint_written(restored=True)` and return `ResumeReport`.

`ResumeReport.bit_identical_expected` is true only when `rng`, `data`, `memory`,
and `optimizers` were all restored with no skips. When they were not, the notes
say "warm restart" explicitly.

### Integration

`RunConfig.resume_from` ([`orchestrator.py:54`](mactf/orchestration/orchestrator.py#L54))
calls `resume_from_last_checkpoint` instead of `CheckpointManager.restore`.
`CheckpointManager` is untouched and remains the low-level writer, so the v1
format is still readable.

### Tests

`tests/logging/*` (records, sinks, async ordering, redaction, signals, sink
failure), `tests/watchdog/*` (policy bounds, every probe, action ordering,
no-mutation), `tests/checkpoints/*` (payload round-trip, verify, skip-corrupt,
resume ordering, bit-identical resume, warm-restart reporting, v1 migration,
prune, adapters, memory format).

---

## 4.10 Peer tree, hierarchical scoring, and hierarchical inference

### Problem

`Peer` ([`peer.py:19`](mactf/coopetition/peer.py#L19)) has no parent, no children,
and no path. `Episode.participants` is keyed by a bare name
([`episode.py:97`](mactf/orchestration/episode.py#L97)), as is
`PhaseContext.peers` ([`phases.py:42`](mactf/orchestration/phases.py#L42)). There
is no tree, no delegation, no synthesis of a parent proposal, and no hierarchical
inference.

Full specification: [10-hierarchical-peers.md](10-hierarchical-peers.md).
Normative rule: [10](10-hierarchical-peers.md) section 1, restated as design
principle P9 in [00](00-scope-and-audit.md) section 4.

### Module: `mactf/coopetition/tree.py`

`NodeKind` (`PEER`, `SUBAGENT`, `MINIAGENT`, `MICROAGENT`, `NANOAGENT`,
`GENERIC`), `PeerNode`, `PeerTree`, `HierarchySpec`, `LevelSpec`. Path grammar with
rejection of `..`, empty segments, orphan and duplicate paths; lexicographic
iteration; four walk orders; `validate()` reporting every structural fault as a
`TreeValidationError` (a `ContractError` subclass, so the API maps it to 422).

`PeerTree.flat(names)` is the compatibility entry point: depth-1 tree whose
`root_paths` are today's peer names.

### Module: `mactf/coopetition/delegation.py` and `synthesis.py`

`DelegationPolicy` with `Broadcast`, `SelectK`, `RouteByRole`, `Confidence`,
`Explicit`; selection recorded per node so starvation is observable.
`SynthesisFunction` with `ConsensusSynthesis` (promoting the existing `consensus`
helper, [`episode.py:162`](mactf/orchestration/episode.py#L162)),
`WeightedConsensus`, `BestByScore`, `LearnedSynthesizer`, `ConcatSynthesis`,
`ExplicitSynthesis`.

### Module: `mactf/scoring/hierarchy.py`

`HierarchicalScorer` scores a tree leaves-first through per-node
`ScoringPipeline` instances, unions child `void_rules`, and produces
`EpisodeScorecard` keyed by path. `RollupStrategy` implementations:
`KappaWeightedRollup` (default), `MeanRollup`, `MinimumRollup`, `WorstCaseRollup`,
`SumRollup`, `ExplicitRollup`.

Non-offsettable propagation: `branch_void` is computed bottom-up and passed to
`compliance.constrain` at every ancestor, so a nano-agent's P1 floors its root
regardless of sibling rewards.

### Determinism fix in the data loader

`TaskSource.next` seeds with `hash((self.seed, epoch, self._counter))`
([`episode.py:63-65`](mactf/orchestration/episode.py#L63-L65)). Integer tuples are
stable today, but a hierarchical run adds node path strings to that tuple, and
string hashing is salted by `PYTHONHASHSEED`. The loader must derive seeds with
`blake2b`, as `10` section 7.4 specifies (`node_seed`). This is a bug fix required
by hierarchy, not a new convention.

### Hierarchical inference

`run_inference(tree, inputs, *, delegation, synthesis, max_depth, budget,
parallel, max_workers, score)` reuses the same tree, delegation, and synthesis
objects as training. Same-level batching when contracts agree, per-node fallback
otherwise with the decision recorded. Budget exhaustion raises `ContractError`
naming the path, never a truncated result. `score=True` emits rule events and can
score with the same pipeline without persisting warning state.

### Integration

`MACTFTrainer(hierarchy=...)` builds the tree recursively, applies finetuning per
node, binds optimizers after freezing, and runs the seven-step epoch from `10`
section 6.2. `Orchestrator` builds `PeerTree.flat(list(peers))` so the existing
loop is a depth-1 case of the same code.

### Tests

`tests/hierarchy/test_paths.py`, `test_tree_build.py`, `test_tree_validate.py`,
`test_tree_walk.py`, `test_flat_equivalence.py`, `test_delegation.py`,
`test_synthesis.py`, `test_rollup.py`, `test_branch_void.py`,
`test_warning_scoping.py`, `test_train_epoch.py`, `test_aggregation_groups.py`,
`test_inference.py`, `test_inference_determinism.py`,
`test_inference_scoring.py`, `test_failure_isolation.py`, `test_prune.py`,
`test_caps.py`.

---

## 4.11 Points scale and cohort normalization

### Problem

Gap 16 and gap 17 in [00-scope-and-audit.md](00-scope-and-audit.md). The library
produces one unbounded number and calls it the score. Three things a framework
claiming a 0-100 rubric must do are absent:

1. No per-rule point value exists. `framework.md` section 9 orders and caps
   severities but leaves magnitudes to the operator, and the operator's decision
   was never written down anywhere in code.
2. The framework's eligibility rules are unenforced. R1 must not be awarded
   alongside a fabrication penalty, R9 requires verified correctness, R10 requires
   genuinely distinct contributions, R3 is capped at `F_cap`, and P19 fires on a
   cohort precision below `P_min`. None of those checks exist, because the docx
   equations sum events without asking whether an event was earned.
3. No cohort statistic exists, so no peer can be told how it did relative to its
   peers, and the aggregator's input range is whatever the equations produced.

### Module: `mactf/scoring/points.py`

```python
MAX_POINTS = 100.0
MIN_POINTS = 0.0

SEVERITY_FRACTIONS = {"S0": 0.0, "S1": 0.05, "S2": 0.15, "S3": 0.4, "S4": 1.0}

DEFAULT_REWARD_WEIGHTS = {
    "R1": 25.0, "R2": 12.0, "R3": 2.0, "R4": 3.0, "R5": 3.0,
    "R6": 5.0, "R7": 5.0, "R8": 6.0, "R9": 3.0, "R10": 8.0,
    "R11": 5.0, "R12": 4.0, "R13": 6.0, "R14": 5.0, "R15": 3.0,
    "R16": 4.0, "R17": 1.0,
}

DEFAULT_PENALTY_WEIGHTS = {
    "P1": 40.0, "P2": 40.0, "P3": 20.0, "P4": 5.0, "P5": 25.0,
    "P6": 6.0, "P7": 6.0, "P8": 5.0, "P9": 15.0, "P10": 15.0,
    "P11": 4.0, "P12": 4.0, "P13": 60.0, "P14": 60.0, "P15": 10.0,
    "P16": 25.0, "P17": 100.0, "P18": 10.0, "P19": 10.0, "P20": 100.0,
    "P21": 100.0, "P22": 100.0, "P23": 12.0,
}

@dataclass(frozen=True)
class RulePoints:
    rule_id: str
    kind: str
    tier: str
    weight: float
    severity_fractions: Mapping[str, float] = field(default_factory=lambda: SEVERITY_FRACTIONS)

@dataclass(frozen=True)
class PointsTable:
    version: str
    rewards: Mapping[str, RulePoints]
    penalties: Mapping[str, RulePoints]
    warning_notional: float = 2.0

    @property
    def reward_total(self) -> float: ...
    def validate(self) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "PointsTable": ...

DEFAULT_TABLE = PointsTable(version="1.0.0", rewards=..., penalties=...)


@dataclass(frozen=True)
class RewardPoints:
    total: float
    rules: tuple[RulePoints, ...]
    denials: tuple[DenialRecord, ...]

@dataclass(frozen=True)
class PenaltyPoints:
    total: float
    saturated: bool
    rules: tuple[RulePoints, ...]

@dataclass(frozen=True)
class WarningPoints:
    count: int
    notional: float
    deduction: float = 0.0

@dataclass(frozen=True)
class PointsBreakdown:
    reward: float
    penalty: float
    warning_notional: float
    gross: float
    final_points: float
    void: bool
    saturated: bool
    rewards: tuple[RulePoints, ...]
    penalties: tuple[RulePoints, ...]
    denials: tuple[DenialRecord, ...]
    table_version: str

    def as_dict(self) -> dict[str, Any]: ...
```

The arithmetic, in one place:

```
reward_points  = sum(w(reward) * f(severity) * 0.5**(occurrence-1))   for earned rewards
penalty_points = sum(w(penalty) * f(severity))                        for applicable penalties
gross          = reward_points - penalty_points
final_points   = 0.0 if void else clamp(gross, 0.0, 100.0)
```

Rewards diminish on repetition; penalties do not. That asymmetry is the whole
anti-farming design: a peer cannot reach 100 by repeating R5, but a peer that
repeats a violation accumulates without relief.

`PointsTable.validate()` enforces, in order: exact 100.0 reward total, one binding
per catalog rule with no extras, non-offsettable penalties at least as large as
the largest reward, and monotone severity fractions with `S0 == 0` and `S4 == 1`.
It raises `ImproperlyConfigured` with the offending key and value, matching the
standard `validate_settings` already holds.

### Module: `mactf/scoring/evidence.py`

`EpisodeEvidence` is the structured form of what the episode actually
demonstrated: task outcome and verification state, each peer's contributions and
their distinctness, refusals and whether an unknown was labelled, flags submitted
versus accepted, remedies offered, partial-versus-refusal status, sandbox and
progress audit results, and the disclosed ambiguities.

`ClaimAudit` runs the claim scan: it separates claims that carry evidence from
claims that do not, and records the count and the ids. It is what makes R14 and
P1 decidable without re-parsing prose at scoring time.

Both are built once, in pipeline step `1a`, and shared by every guard. A guard
must not re-derive evidence from the ledger: a second derivation is a second
opinion, and the whole point of `EpisodeEvidence` is that all 40 guards agree
about the same facts.

### Module: `mactf/scoring/guards.py`

```python
@dataclass(frozen=True)
class GuardVerdict:
    earned: bool
    reason: str | None = None
    missing: tuple[str, ...] = ()
    cap_applied: bool = False

@dataclass(frozen=True)
class RuleBinding:
    rule_id: str
    guard: Callable[[EpisodeEvidence, ClaimAudit], GuardVerdict]
    requires: tuple[str, ...] = ()

class GuardRegistry:
    def resolve(self, rule_id: str, evidence: EpisodeEvidence,
                audit: ClaimAudit) -> GuardVerdict: ...
    def register(self, binding: RuleBinding) -> None: ...
    def bindings(self) -> Mapping[str, RuleBinding]: ...
```

40 bindings, one per catalog rule. The three-way outcome is deliberate:
`earned=False, reason="denied"` means the evidence contradicts the credit;
`earned=False, missing=(...)` means the guard could not decide. Collapsing the
second into the first would tell a peer it failed when the truth is that nobody
supplied the record, and that distinction is what makes a denial actionable.

The guard table, its per-rule justifications, and the denial semantics are in
[11-points-and-normalization.md](11-points-and-normalization.md) section 4.

### Module: `mactf/scoring/normalize.py`

```python
@dataclass(frozen=True)
class NormalizationPolicy:
    enabled: bool = True
    target_mean: float = 62.0
    shrink: float = 0.35
    exclude_void: bool = True
    min_cohort: int = 2
    signal_scale: float = 50.0
    clip: tuple[float, float] = (MIN_POINTS, MAX_POINTS)

class CohortNormalizer:
    def normalize(self, points: Mapping[str, float], *, episode: int = 0,
                  void: Mapping[str, bool] | None = None) -> NormalizationReport: ...
```

```
calibrated = clamp(target_mean + shrink * (points - mean), 0, 100)
signal     = (calibrated - target_mean) / signal_scale
```

Stateless. Kappa already owns the temporal feedback, and a second controller with
memory would fight it. Void peers are excluded from the statistics before the mean
is taken, then pinned to `0.0`; including them would let one fabrication push
everyone else's calibrated score up.

### Settings

```python
SCORING["POINTS"] = {
    "TABLE_VERSION": "1.0.0",
    "SEVERITY_FRACTIONS": {...},
    "WARNING_NOTIONAL": 2.0,
    "REWARD_WEIGHTS": {...},        # must sum to 100
    "PENALTY_WEIGHTS": {...},
}
ENGINE["F_CAP"] = 3
ENGINE["P_MIN"] = 0.5
ENGINE["P12_NOISE_RATIO"] = 0.5
ENGINE["R9_STEP_FACTOR"] = 1.25
COOPETITION["SIGNAL_SOURCE"] = "points"
COOPETITION["NORMALIZATION"] = {"ENABLED": True, "TARGET_MEAN": 62.0, "SHRINK": 0.35, ...}
```

`F_CAP` and `P_MIN` are named in document 02 section 3.5 as `SCORING.P_MIN`; the
key moves to `ENGINE` here because both are episode-level engine thresholds rather
than weights, and 02 is amended to point at this section.

### Integration

- `RewardScorer.points`, `PenaltyScorer.points`, `WarningScorer.points` — 07
  sections 4.1, 5.1, 6.1.
- `ScoringPipeline` steps `1a`, `3a`, `4a` and `score_points` — 07 sections 7 and
  7.2.
- `ScoringPipeline.score_episode_normalized` — 07 section 7.1.
- `PhaseContext.signal_by_peer(source=)` and `COOPETITION.SIGNAL_SOURCE` — 08.
- `ScorecardSerializer.points`, `ScorecardV2Serializer`, three routes — 02
  sections 2.2.1, 3.1, 3.3, 4.3.1.
- `Points`, `Normalization`, `scores` namespace — 03 sections 4.2, 8, 9.1.
- Migration 3 adds `scorecards.points`, `scorecards.points_table_version`,
  `episodes.normalization`, `aggregations.signal_source`.

### Tests

| File | Covers |
|---|---|
| `tests/scoring/test_points_table.py` | Weights sum to 100, one binding per rule, non-offsettable floor rule, validation messages |
| `tests/scoring/test_guards.py` | One row per rule, plus the named discriminations (R1 vs P1, R3 cap, R9 unverified, R10 concatenation, R11 no-remedy, P16 disclosed ambiguity, P19 cohort precision) |
| `tests/scoring/test_points_monotone.py` | M1-M4 monotonicity, void pinning, the `final_score > 0` / `final_points == 0` case |
| `tests/scoring/test_points_equivalence.py` | `final_score`, `floor`, `is_void`, `continuous` identical to the pre-change oracle; import independence both ways |
| `tests/scoring/test_normalize.py` | The worked example, `shrink` extremes, clipping, degenerate cohorts |
| `tests/scoring/test_normalize_invariants.py` | N1-N6, permutation invariance, `preserves_order` wired |
| `tests/contrib_api/test_points_routes.py` | Three routes, 404/409 cases, scale completeness |
| `tests/contrib_api/test_scorecard_v2.py` | Header and query negotiation, 406 body, representation agreement |
| `tests/training/test_signal_source.py` | Entropy convergence under both sources, kappa rescaling, resume persistence |
| `tests/client/test_points_parity.py` | `from_dict` round-trips, both transports agree, v2 on both |

### Sequencing

```
points.py -> evidence.py -> guards.py -> scorer points() methods
          -> pipeline composition -> equivalence gate green
          -> normalize.py -> signal source -> migration 3 -> routes -> client
```

The equivalence gate closes before `normalize.py` because normalization compresses
`final_points` and would happily compress a wrong one. Migration 3 closes before
the routes because a route that reads a column must not ship before the column
exists.

## 4.12 Framework integration of the service layer

Gap 18 is not a capability gap; it is the observation that a service layer built
beside the framework contradicts machinery the framework already has. This
section closes it by using that machinery, and it is the only section in this
document whose deliverable is structure rather than behaviour. Nothing a
scorecard contains changes.

### Module: `mactf/http/`

The transport, stdlib only. Full specification in
[12-http-core.md](12-http-core.md); the surface, in one place:

```python
from mactf.http import Request, Response, JsonResponse, StreamingResponse, ProblemResponse
from mactf.http.urls import path, re_path, include, resolve, reverse
from mactf.http.views import View
from mactf.http.handler import get_wsgi_application
from mactf.http.serializers import Serializer, ModelSerializer, ValidationError
```

It may import `mactf.exceptions` and nothing else from `mactf`. Seven modules:
`request`, `response`, `urls` (+ `router`), `views`, `middleware`, `handler`,
`serializers`, plus `exceptions` for the RFC 9457 mapping, `openapi` for the
schema, and `asgi` for the optional adapter.

### Module: `mactf/contrib/api/`

The app. What it must contain, and why each file exists rather than being inlined
into a factory function:

| File | Contents | Why it is a file |
|---|---|---|
| `apps.py` | `ApiConfig(name="mactf.contrib.api", label="api", default_auto_field=...)`, `urls_module = "mactf.contrib.api.urls"`, `ready()` imports views, registers checks, connects signal receivers | `ready()` is the framework's app hook; using it is the whole point of Gap 18 |
| `urls.py` | `app_name = "api"`, `urlpatterns` with `name=` on every route from doc 02 | One route table, used by `resolve()`, `reverse()`, the OpenAPI generator, and the contract test |
| `views.py` + `views/` | one `View` subclass per route group | A view returns a `Response` and calls the engine; it computes nothing |
| `serializers.py` | every wire model from doc 02 | The contract lives with the app that publishes it |
| `auth.py` | bearer tokens, scopes, `Principal` | `API.TOKENS`/`API.ALLOW_ANONYMOUS` are app settings |
| `checks.py` | `mactf.api.E001` (no tokens and anonymous disallowed), `W002` (`ROOT_URLCONF` missing or not including the app), `E003` (duplicate route name), `E004` (unknown scope or malformed token digest) | Checks are how an opt-in app reports misconfiguration without crashing |
| `signals.py` | `app_ready` receiver, run lifecycle receivers | An app may listen; nothing in core may know this app exists |
| `management/commands/serve.py` | `mactf serve` on `wsgiref` | Commands are discovered from apps, so `serve` is absent when the app is |

`ApiConfig.ready()` is also where the app-scoped `ScoringPipeline` and
`RunManager` are constructed and hung on the config, which is what makes them
per-app rather than process globals.

### Module changes in `mactf/apps.py`, `mactf/conf.py`, `mactf/db.py`, `mactf/management/`

Four small additive changes, each with its own test. Specification in
[13-app-integration.md](13-app-integration.md).

```python
# mactf/apps.py
class AppConfig:
    urls_module: str | None = None
    migrations_module: str | None = None
    checks_module: str | None = None
    def get_urls(self) -> list[URLPattern] | ModuleType: ...
    def get_commands(self) -> dict[str, str]: ...
    def get_migrations(self) -> list[ModuleType]: ...
```

```python
# mactf/conf.py DEFAULTS gains
"ROOT_URLCONF": None,
"APPEND_SLASH": True,
"MIDDLEWARE": ["mactf.http.middleware.RequestIdMiddleware",
               "mactf.http.middleware.ExceptionMiddleware"],
"HTTP": {"STREAM_WORKERS": 16, "SSE_HEARTBEAT": 15.0, "MAX_BODY_BYTES": 8 << 20},
"API": {"ALLOW_ANONYMOUS": True, "TOKENS": {}, "BASE_PATH": "api/v1",
        "EVENT_BUFFER": 1024, "MAX_CONCURRENT_RUNS": 1},
```

`mactf/management/base.py:17` changes `COMMAND_PACKAGES` from a literal to
`core packages + [f"{app.name}.management.commands" for app in apps.get_app_configs()]`.
`mactf/db.py` gains the `mactf_migrations(app, version, applied_at)` table and
`MIGRATIONS` keyed by app label, with `core` first and apps in label order.

### Module: `mactf/client/`

The client, in core, because it depends only on L3 and on `from_dict`.

```python
class MACTFClient:
    def __init__(self, base_url: str | None = None, *,
                 transport: Literal["auto", "in_process", "http"] = "auto",
                 settings: Settings | None = None, timeout: float = 30.0,
                 token: str | None = None) -> None: ...
```

`InProcessTransport` builds a WSGI environ and calls
`get_wsgi_application()`; `HTTPTransport` sends the same method, path, headers,
and body over `urllib.request`. Both hand the raw status, headers, and body to
the same `Response`-parsing code and then the same `from_dict` call. That shared
tail is what makes parity a property of the design rather than a test that could
pass by accident.

`mactf.testing.Client` is the same in-process path with a test-shaped API
(`.get()`, `.status_code`, `.json()`), replacing `fastapi.testclient.TestClient`
in every fixture.

### The `from_dict` half

Nine classes gain `from_dict`, and it is a strict inverse of `to_dict`:

| Class | Module |
|---|---|
| `PeerScorecard` | `mactf/scoring/scorecard.py` |
| `EpisodeScorecard` | `mactf/scoring/scorecard.py` |
| `ContinuousScore` | `mactf/scoring/continuous.py` (required by `PeerScorecard`) |
| `AggregationReport` | `mactf/coopetition/aggregation.py` |
| `CapabilityReport` | `mactf/sandbox/base.py` |
| `ExecResult` | `mactf/sandbox/base.py` |
| `ComplianceResult` | `mactf/compliance.py` |
| `PointsBreakdown` | `mactf/scoring/points.py` |
| `NormalizationReport` | `mactf/scoring/normalize.py` |

Document 13 section 5 is the normative list and explains how each is derived;
this table exists so the phase-A0 reader knows the scope without leaving the
document. Each raises `KeyError` on a missing required field rather than
defaulting to `None`. That is deliberate: a client that silently fills a missing
score with `None` reports a peer as unscored when the truth is that the server
sent a malformed body, and the failure surfaces months later.

### Integration

- `AppConfig` extensions, `command_packages()`, app-scoped migrations, new
  `DEFAULTS` keys — [13](13-app-integration.md) sections 2-5.
- Transport surfaces and layering rule — [12](12-http-core.md).
- Every route becoming a `name=`d entry in `mactf/contrib/api/urls.py` — 02
  section 3.
- Client namespaces returning framework objects — 03 sections 3-6.
- `docs/plans/01` section 2 for the module graph and section 3 for the
  threaded-server concurrency model.

### Tests

| File | Covers |
|---|---|
| `tests/http/test_layering.py` | Criterion 39: an AST walk proves `mactf/http` imports no engine module |
| `tests/http/test_request.py`, `test_response.py` | Criterion 40, 42: environ parsing, chunked and empty bodies, JSON, streaming, problem+json |
| `tests/http/test_urls.py` | Criterion 41: `resolve()` and `reverse()` for every route and kwargs combination |
| `tests/http/test_middleware.py` | Criterion 43: order in and out, exception translation, request-id echo |
| `tests/http/test_serializers.py` | Criterion 44: round-trip per wire model, unknown field rejected |
| `tests/http/test_handler.py` | Criterion 45: `ROOT_URLCONF` and `MIDDLEWARE` read lazily, cached per settings object |
| `tests/http/test_openapi.py` | Criterion 46: generated schema equals the doc 02 route table |
| `tests/contrib_api/test_uninstalled.py` | Criterion 47: removing the app from `INSTALLED_APPS` breaks nothing else |
| `tests/contrib_api/test_checks.py` | Criterion 48: checks registered only when installed, `E001` fires on the misconfiguration |
| `tests/contrib_api/test_serve.py` | Criterion 49: `wsgiref` serves `/health` with no third-party HTTP package |
| `tests/contrib_api/test_urls.py` | Criterion 50: `include()` composition and `reverse("api:rule-list")` |
| `tests/contrib_api/test_migrations.py` | Criterion 51: app-scoped migration order and idempotence |
| `tests/client/test_in_process.py` | Criterion 52: full run with no socket, returned classes are the framework's |
| `tests/client/test_http.py`, `test_parity.py` | Criterion 53: byte-identical bodies and identical objects across transports |
| `tests/client/test_no_parallel_models.py` | No locally defined dataclass shadows an engine class |
| `tests/testing/test_client.py` | Criterion 54: no server, `status_code`/`json()`, no settings leak between tests |

### Sequencing

```
4.12 http layer ──> api app (urls, views, serializers) ──> serve ──> client
                    └─> checks, app-scoped migrations
                           └─> from_dict on the nine classes (parallel, no deps)
```

The `from_dict` work is independent of the transport and can land first; the
client cannot. `mactf/contrib/api` cannot land before `mactf/http` exists, and
`serve` cannot land before command discovery from apps works, which is why
`mactf/management/base.py` is edited before the api app, not after.



```
4.3 evaluator ─┬─> 4.4 persistence (adjudication tables reference rule ids)
               └─> 4.6 pool (needs R10/R11/R16 to be producible)
4.1 data ───────> 4.2 bootstrap (bootstrap needs a real source)
4.5 guest agent  (independent)
```

4.3 first, because 4.4 and 4.6 are meaningless without it. 4.1 before 4.2 because
`BootstrapTrainer` consumes the source protocol. 4.5 anywhere.

The extension gaps form a separate phase, A0, that depends on none of the above:

```
4.7 scorer split ──> 4.10 tree and hierarchy
4.7 functions ─────> 4.8 trainer
4.7, 4.10 ─────────> 4.8 trainer wiring
4.8 trainer ───────> 4.9 logger, watchdog, checkpoint wiring
4.10 tree ─────────> 4.9 topology in the checkpoint payload
```

Within 4.7 the order is base, reward, penalty, warning, pipeline, gate, then
functions. Within 4.10 it is tree, delegation, synthesis, hierarchy, flat gate,
then inference. Both gates are prerequisites for 4.8.

4.11 depends on 4.7 and not on 4.10:

```
4.7 pipeline ──> 4.11 points.py, evidence.py, guards.py
             ──> 4.11 scorer points() methods
             ──> 4.11 composition ──> 4.11 equivalence gate
                                    ──> 4.11 normalize.py
             ──> 4.11 migration 3 ──> routes ──> client
```

The reverse is not true: 4.7's existing equivalence gate does not depend on 4.11,
which is why 4.11 can land after `hierarchy.py` without reopening anything. What
4.11 must not do is land *before* 4.7's gate is green, because then a points bug
and a split bug would arrive together.

`4.10` and `4.11` both touch `mactf/scoring/hierarchy.py`'s caller: 4.10 adds
`branch_void` to `score_peer`, 4.11 reads it in step `4a`. 4.10 first, so the
argument exists before the reader does.

A0 and A can run in either order or concurrently; they touch disjoint files
except `mactf/scoring/__init__.py` and `mactf/coopetition/__init__.py`, which A0
owns.

## 6. What Phase A explicitly does not change

- `mactf/rules/` - catalog, severity, precedence are untouched. The evaluator
  reads them; nothing writes them.
- `mactf/compliance.py` - the floor arithmetic is not modified. 4.6 reports
  alongside it, and 4.10 passes `branch_void` into it.
- `mactf/coopetition/aggregation.py` - weight construction is untouched.
  `KappaSoftmaxAverage` in 4.8 calls it; it does not reimplement it.
- `docs/framework.md` and `docs/framework.md.bak` - read-only.

Amendments to that list, required by A0:

- **`mactf/rl/` is no longer untouched.** The free functions are unchanged and
  remain the implementation, but `mactf/rl/functions/` is added and
  `mactf/rl/__init__.py` gains exports. `compute_loss`, `policy_loss`,
  `generalized_advantage`, and `value_loss` keep their signatures and their tests.
- **`mactf/orchestration/phases.py` changes.** `local_update` is relocated behind
  `TrainerProtocol`. The arithmetic is identical, and the equivalence gate in 4.8
  proves it, but the module is edited.
- **`mactf/orchestration/episode.py` changes.** The seeding in `TaskSource.next`
  moves to `blake2b` (4.10), and `TaskSource` gains counter restore for resume.
- **`mactf/orchestration/checkpoint.py` is extended, not replaced.**
  `CheckpointManager` keeps its API and its tests; `checkpoint_v2.py` adds
  `MACTFCheckpoint` above it. The v1 format stays readable.
- **`mactf/coopetition/peer.py` changes.** `build_peer` gains an
  `optimizer_factory` parameter defaulting to the current Adam, and `Peer.to`
  delegates to `MACTFOptimizer.move_to` when attached. Default behaviour is
  byte-identical.
- **`mactf/scoring/__init__.py` changes.** It gains the new exports and keeps
  `Scorer` resolvable both as the ABC and, through the shim, as the pipeline.
- **`mactf/telemetry.py` is unchanged.** `MACTFLogger` wraps it; it does not
  replace `configure_logging`, `JsonFormatter`, or `log_event`.

Amendments required by 4.11:

- **`mactf/scoring/scorecard.py` gains a field, not a change.**
  `PeerScorecard` carries `points: dict | None` and `branch_void` stays as 4.10
  added it. Every existing field keeps its meaning, which is what makes criterion
  35 achievable.
- **`mactf/coopetition/aggregation.py` is untouched.** `AggregationReport` gains
  `signal_source` and `normalization` as additive defaults; `compute_weights` and
  the simplex projection are not edited. The vector handed to them changes, and
  that is the entire difference.
- **`mactf/orchestration/phases.py` gains a parameter.**
  `PhaseContext.signal_by_peer` grows `source="points"`, defaulting to the settings
  value through `aggregate`. Under `source="final_score"` the function returns
  what it returns today.
- **`mactf/db.py` gains migration 3**, additive columns only. No existing column is
  retyped, no existing row is rewritten, and `migrate` does not backfill points:
  the `mactf rescore --points` command does that, explicitly and per run.
