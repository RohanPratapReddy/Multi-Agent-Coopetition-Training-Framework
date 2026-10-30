---
title: Scope and Audit
status: approved-for-implementation
version: 1.0.0
applies_to: mactf >= 0.1.0
---

# 00. Scope and Audit

## 1. Purpose

MACTF today is a **library plus a CLI**. `mactf run` trains a swarm, writes
scorecards to sqlite, and exits. That is a complete developer workflow but not a
usable product surface: there is no way to start a run from another process,
watch it live, feed an external evaluator's judgments into the rule engine, or
drive the framework from a notebook or a service in another language.

This plan adds:

1. A **transport layer** (`mactf/http/`) built on the standard library: `Request`,
   `Response`, `path`/`include`/`reverse`, `View`, middleware, `WSGIHandler`, and
   declarative `Serializer`s. Nothing in it is a third-party dependency, and
   nothing in it knows what a peer is.
2. A **service layer as an installed app** (`mactf.contrib.api`) selected by
   `INSTALLED_APPS` and mounted by `ROOT_URLCONF`, exactly as
   `django.contrib.admin` is. Removing the entry removes the HTTP API and breaks
   nothing else.
3. A **Python client in core** (`mactf.client`) with two transports — in-process
   dispatch through `ROOT_URLCONF`, and `urllib` against `mactf serve` — both
   returning the framework's own dataclasses. There is no parallel model family
   and no separate distribution.
4. The **internal implementations** that make (1)-(3) honest rather than
   decorative - a REST endpoint that returns a scorecard built by a no-op judge
   is worse than no endpoint at all.
5. A **wrapper layer** over the library's own primitives, so an orchestrator can
   be composed from parts instead of edited: `Scorer` subclasses
   ([07](07-scorers-and-functions.md)), `ValueFunction` / `AdvantageFunction` /
   `LossFunction` ([07](07-scorers-and-functions.md)), `MACTFTrainer` /
   `MACTFOptimizer` / `PeerStrategy` ([08](08-trainer-optimizer-strategies.md)),
   `MACTFLogger` / `MACTFWatchdog` / `MACTFCheckpoint`
   ([09](09-logging-watchdog-checkpoint.md)), and arbitrary-depth peer trees with
   hierarchical inference ([10](10-hierarchical-peers.md)).

Item 5 is not a rewrite of the engine. It is the seam that lets a user replace
one part (a reward function, an optimizer, a training step) without forking the
phase module, and it is specified because today there is no such seam - gaps
9-17 in section 3.

Items 1-3 are one design, not three packages. The framework already ships the
machinery this needs — `mactf/apps.py` with `INSTALLED_APPS` and `ready()`,
`mactf/checks.py`, `mactf/management/base.py`, `mactf/db.py` migrations,
`mactf/signals.py` — and the service layer uses it rather than duplicating it.
That is what Gap 18 records, and why P12 below exists.

## 2. Sources

| Source | Availability | Used for |
|---|---|---|
| `docs/framework.md` | full | All scoring, precedence, warning, aggregation semantics |
| `docs/Designing_Multi_Agent_Incentive_Frameworks.docx` | full (extracted) | Continuous equations, kappa controller, loss/advantage forms, QEMU design |
| `docs/gemini_conversation_28459b25aa1c1b46.docx` | full (byte-identical to the above) | Confirmation of the 10-turn design conversation |
| `https://gemini.google.com/app/28459b25aa1c1b46` | full, via the docx export | Same |
| `https://gemini.google.com/app/7ac3731679b9b5fd` | **not accessible** | Unauthenticated fetch returns the Gemini landing page. No content used. |
| `https://gemini.google.com/app/8f7dcbb2ed0643e0` | **not accessible** | Same. No content used. |

The two inaccessible threads are not silently assumed to contain nothing. They
are excluded, and if they carry requirements, those requirements are currently
unrepresented in this plan. Exporting them into `docs/` the way the third was
exported is the only way to fold them in; see section 9.

## 3. Verified gaps in the current library

Every claim below was checked by reading the source, not the README. Line
numbers refer to the state of the tree at the time of writing.

Gaps 1-8 are capability gaps closed in phase A and covered by documents
[04](04-gap-closures.md), [02](02-api-contract.md), [12](12-http-core.md), and
[13](13-app-integration.md). Gaps 9-17 are extension gaps closed in phase A0 and
covered by documents [07](07-scorers-and-functions.md),
[10](10-hierarchical-peers.md), [08](08-trainer-optimizer-strategies.md),
[09](09-logging-watchdog-checkpoint.md), and
[11](11-points-and-normalization.md). Gap 18 is the integration gap that this
revision exists to close.

### Gap 1 - Bootstrap is a documented no-op

[mactf/orchestration/phases.py:67](mactf/orchestration/phases.py#L67)

```python
def bootstrap(context: PhaseContext, *, enabled: bool = True) -> dict[str, Any]:
    signals.phase_started.send(sender=context, phase="bootstrap")
    summary: dict[str, Any] = {"enabled": enabled}
    if not enabled:
        signals.phase_finished.send(sender=context, phase="bootstrap", summary=summary)
        return summary

    # Priming is architectural here: peers start from the same weights and the
    # reference is a frozen snapshot, which is what rule memorization needs.
    signals.phase_finished.send(sender=context, phase="bootstrap", summary=summary)
    return summary
```

`TRAINING["BOOTSTRAP"]` defaults to `True` and trains nothing. The docx Phase 2
specifies behavioural cloning over ideal multi-agent transcripts; that is not
implemented. An API that reports "bootstrap: enabled" while performing zero
gradient steps would be a false statement about the run.

### Gap 2 - No dataset loader

[mactf/orchestration/episode.py:34](mactf/orchestration/episode.py#L34)
`TaskSource` generates synthetic Gaussian tensors and an argmax-of-projection
target. `training/datasets/truth.jsonl`, `ambiguity.jsonl`, and
`interaction.jsonl` exist on disk with real content:

```
{"id": "truth-001", "claim": "2 + 2 = 4", "evidence": "arithmetic", "verified": true}
{"id": "ambiguity-001", "prompt": "What is the exact number of fish in the Atlantic right now?", "expected": "admitting_ignorance"}
{"id": "interaction-001", "stage": "proposal", "peers": 3, "expects": ["R1", "R2"]}
```

Nothing reads them. `PATHS["DATASETS"]` is referenced only in the defaults table
in [mactf/conf.py:52](mactf/conf.py#L52). Each file holds two records, so they
also demonstrate that no loader exists which can handle a corpus: there is no
code that validates, batches, or iterates them.

### Gap 3 - Most rules are unreachable

[mactf/scoring/judge.py](mactf/scoring/judge.py) records exactly R1, R2, and R14
from measured accuracy and calibration. There is no path for an external
evaluator to submit evidence for any other rule. Consequently 14 reward rules and
all 23 penalty rules can never fire in a normal run:

- Rewards never emitted: R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R15,
  R16, R17
- Penalties never emitted: P1-P23, without exception

`redteam.py` injects P1, P7, and P20 as hand-built ledgers, which proves the
scorer works but is not an ingest path. The README states this plainly under
"Known limits":
"Subjective rules (R10 synthesis, P13 collusion, P16 loophole) require an
external evaluator to supply its own ledger events." There is no API for that
supply.

### Gap 4 - Warnings are not persisted

[mactf/db.py](mactf/db.py) defines `MIGRATIONS[1]` with `mactf_schema`, `runs`,
`episodes`, `rule_events`, and `checkpoints`. There is no `warnings` table and no
`adjudications` table. `Scorer.registries` is in memory;
`WarningRegistry.snapshot()` exists and is written into checkpoints only. A
`--resume` therefore restores warnings only when a checkpoint is used, and an
adjudicator reversal (W5) has nowhere to be recorded at all.

### Gap 5 - QEMU exec is honest but non-functional

[mactf/sandbox/qemu.py:317](mactf/sandbox/qemu.py#L317) returns exit code 127 with
an explanatory message when no serial channel exists, and the channel path needs
a guest listener that is not shipped. The host side
([mactf/sandbox/channel.py](mactf/sandbox/channel.py)) is complete: `Frame`,
`FrameReader`, `SocketChannel.request`. The cloud-init writer
([mactf/sandbox/provision.py:116](mactf/sandbox/provision.py#L116)) produces a
`runcmd` that only touches `/run/mactf.ready`.

An API endpoint `POST /sandbox/exec` that always returns 127 would be a stub.

### Gap 6 - Cooperative pool is inert

`COOPERATIVE_POOL: True` appears in
[training/settings.py:112](training/settings.py#L112) and in
[mactf/conf.py](mactf/conf.py) defaults. No module reads it.
`PeerScorecard` ([mactf/scoring/scorecard.py:22](mactf/scoring/scorecard.py#L22))
reports `final_score` only. framework.md 9.2 requires the competitive and
cooperative components to be "reported separately", and 9.1/9.4 make R10, R11,
and R16 the cooperative awards.

### Gap 7 - Run execution is blocking

[`Orchestrator.run`](mactf/orchestration/orchestrator.py#L191) runs the epoch
loop inline and returns a dict. There is no cancellation token, no progress
callback, and no way for a caller to observe an epoch before the whole run
finishes. Serving HTTP requires all three.

### Gap 8 - No service or client surface

`mactf/__init__.py` exports `__version__` and `VERSION`. There is no `mactf.http`,
no `mactf.contrib.api`, no `mactf.client`, and no `mactf serve` command
(`mactf/management/commands/` contains check, doctor, inspect, migrate, redteam,
run, sandbox, shell, startpeer, startproject).

The narrower statement matters: the *framework* has everything a service layer
needs except the transport. `INSTALLED_APPS`, `AppConfig.ready()`, app-registered
system checks, signals, versioned migrations, and a management-command loader all
exist and work. What does not exist is a `Request`, a `Response`, a URL conf, or
anything that turns a view into a WSGI response.


### Gap 9 - Scoring is one monolithic class

[`Scorer`](mactf/scoring/pipeline.py#L30) is a single dataclass with four fields
that advances the warning registry, computes the reward, computes the penalty,
applies compliance, and writes the scorecard, all inside `score_peer`
([pipeline.py:53-139](mactf/scoring/pipeline.py#L53-L139)). There is no seam for a
custom reward function, no way to replace the penalty formula without forking
compliance, and no way to score a nested peer at all. Verified by reading every
line of `mactf/scoring/`: no `RewardScorer`, `PenaltyScorer`, or
`WarningScorer` exists.

### Gap 10 - No trainer abstraction

`local_update` ([phases.py:142-224](mactf/orchestration/phases.py#L142-L224))
hardcodes the policy-gradient step: it calls `state_targets_for`, builds a policy
from logits, masks advantages, calls `compute_loss`, and steps. A user with a
supervised objective, a Hugging Face `Trainer` loop, or a PEFT fine-tune cannot
plug in without editing the phase module. There is no `TrainerProtocol`, no
custom-step parameter, and no callback.

### Gap 11 - No optimizer abstraction

`build_peer` constructs `torch.optim.Adam` unconditionally
([peer.py:103-107](mactf/coopetition/peer.py#L103-L107)), and `Peer.to` rebuilds
Adam after a device move ([peer.py:41-47](mactf/coopetition/peer.py#L41-L47)),
discarding whatever optimizer the caller had. There is no `scipy.optimize` path,
no per-parameter-group routing, and no way to express that a LoRA adapter and a
task head want different learning rates.

### Gap 12 - No run-scoped structured logging

[`configure_logging`](mactf/telemetry.py#L39) configures the root `mactf` logger
with two formatters. There is no run id on a record, no epoch or step record
type, no bounded in-memory tail for an API to read, and no sink abstraction. The
orchestrator logs formatted strings
([orchestrator.py:322-328](mactf/orchestration/orchestrator.py#L322-L328)), so
nothing downstream can parse a run's history without regex.

### Gap 13 - No health monitoring

`should_rollback` ([checkpoint.py:150](mactf/orchestration/checkpoint.py#L150))
exists and is called once, after aggregation, from
[orchestrator.py:287](mactf/orchestration/orchestrator.py#L287). Nothing probes
for NaN loss, gradient explosion, throughput collapse, memory growth, stale
progress, or a dead subtree. A run that diverges produces a low score, not an
incident.

### Gap 14 - Checkpoints are not resumable in the strict sense

`CheckpointManager.save` writes
([checkpoint.py:64-78](mactf/orchestration/checkpoint.py#L64-L78)): run id,
episode, runtime dict, kappa, entropy, score, peer states, reference state,
controller state, warnings. It does not write RNG state, data position, scheduler
state, AMP scaler state, adapter separation, buffers, memory format, or any
integrity manifest. `restore` therefore resumes weights and kappa but not the
random stream or the task stream, so a "resumed" run is a warm restart that
diverges from the uninterrupted run within one epoch. There is no `verify`, no
checksum, and no fallback when the newest file is truncated.

### Gap 15 - Peers are flat

`Peer` ([peer.py:19](mactf/coopetition/peer.py#L19)) has no children, no parent,
and no path. `Episode.participants` is keyed by a bare name
([episode.py:97](mactf/orchestration/episode.py#L97)), as is
`PhaseContext.peers` ([phases.py:42](mactf/orchestration/phases.py#L42)). There is
no tree, no delegation, no synthesis of a parent proposal, and no hierarchical
inference. A user who wants `peer -> subagents -> miniagents -> ...` must build
it outside the framework and loses scoring, aggregation, and checkpointing.

### Gap 16 - There is no 0-100 score, and the docx net cannot become one

`ContinuousScore.net` ([continuous.py:102](mactf/scoring/continuous.py#L102)) is
unbounded in both directions and gains magnitude with the number of events.
`compliance.constrain` ([compliance.py:79](mactf/compliance.py#L79)) subtracts the
non-offsettable penalties a second time as an absolute floor
([compliance.py:101-104](mactf/compliance.py#L101-L104)) and clamps only at
`value_cap` (`1e4` by default). There is no ceiling near 100, no normalization,
and no rubric. The `-5.0` a P1 episode reports is the *penalty lambda*, not a
floor: the floor is `-abs(non_offset)`, so it scales with the configured severity.

The framework's own text asks for more than this. `docs/framework.md` section 9
gives an ordered severity ladder with per-tier caps, section 9.5.3 caps flags per
peer per episode (`F_cap`), and section 9.5.4 sets a precision bar (`P_min`), but
none of those three quantities is computed anywhere in the library today. A peer
that fires a T1 penalty and a peer that fires forty T4 penalties can produce the
same `net`, and nothing in the output says which happened.

The naive fix, rescaling `net` onto 0-100, is wrong for the reason P11 states: it
would make the rubric's value change whenever the equation changed, and it cannot
express a void episode any differently from a very bad one. What is missing is a
second, independently derived score with its own table, its own guards, and its own
arithmetic — which is why this is a gap and not a tweak.

### Gap 17 - Nothing calibrates a cohort, and nothing bounds the aggregation input

`aggregate` feeds `FinalScore`-derived signals directly into
`softmax(kappa * score)`
([phases.py:226-263](mactf/orchestration/phases.py#L226-L263),
[aggregation.py:159-168](mactf/coopetition/aggregation.py#L159-L168)). There is no
step that puts peers on a comparable footing, and the signal's range is whatever
the equations produced that epoch. Three consequences are observable today:

- A cohort whose peers differ by 60 points in `net` produces a softmax that is
  effectively one-hot, so the losing peers receive almost no gradient.
- `AdaptiveKappa` compensates by raising `kappa`, which is the correct response to
  a signal that is too flat and the wrong response to a signal that is too wide;
  the controller cannot distinguish the two because it only sees entropy.
- There is no way to ask "how did this peer do relative to its cohort" without
  recomputing the statistics by hand from a list of scorecards.

Nothing in the library computes a cohort mean, a spread, a percentile, or a
calibrated score. `AggregationReport` reports the weights that resulted, not the
inputs that produced them, so a user cannot separate "this peer was scored low"
from "this peer was weighted low by the aggregator".

### Gap 18 - The planned service layer would ignore the framework's own app system

This gap is about the plan as it stood before this revision, and it is the reason
this revision exists.

The previous revision specified two packages beside the framework:
`mactf/api/` (FastAPI, behind an `[api]` extra) and `mactf/sdk/` (httpx, behind an
`[sdk]` extra, with its own frozen dataclass family in `models.py`). Both would
have to re-implement machinery the framework already ships, and in four specific
ways they would contradict it:

| Framework machinery | What the old plan did instead |
|---|---|
| `INSTALLED_APPS` + `AppConfig.ready()` ([mactf/apps.py:17](mactf/apps.py#L17), [:79](mactf/apps.py#L79)) | A `create_app` factory imported by the entry point; nothing selectable, nothing removable |
| App-registered system checks ([mactf/checks.py:67](mactf/checks.py#L67)) | No checks; a misconfigured API started happily |
| `COMMAND_PACKAGES` discovery ([mactf/management/base.py:17](mactf/management/base.py#L17)) | `serve` installed in core, so a bare install shipped a server for an app it did not have |
| Versioned migrations ([mactf/db.py:20](mactf/db.py#L20)) | API-adjacent tables with no migration owner |
| `mactf/signals.py` `app_ready` ([mactf/signals.py](mactf/signals.py)) | Nothing listened; the app system was decorative |

Three symptoms follow directly:

1. **Unremovable.** Removing the API required editing the entry point, not
   settings. A `ImportError` was the only signal that a project had uninstalled
   it, which is the opposite of how `django.contrib.admin` behaves.
2. **Duplicate models.** `mactf/sdk/models.py` would define `Scorecard`,
   `Aggregation`, `Capability`, and `ExecResult` again, with `from_dict`
   hand-maintained against `to_dict`. That is the drift P2 exists to prevent, and
   the drift would be silent: a field added to `PeerScorecard.to_dict` with no
   client-side change is a `None` in a dashboard six months later.
3. **Two request paths that were not the same path.** `LocalBackend` was
   specified to call the engine directly, and `RemoteBackend` to speak HTTP. The
   parity suite compared their *results*, so a serializer bug that only affected
   one field would pass every test if the local path never serialized.

The fix is not smaller packages; it is the arrangement the framework already
implies: a stdlib transport layer, a service layer that is an app, and a client
in core. Documents [12](12-http-core.md) and [13](13-app-integration.md) specify
it, and P12 states the principle.


## 4. Design principles

These constrain every decision in the following documents.

**P1 - The library is the only source of truth.** The API computes nothing. It
calls `Scorer.score_peer`, `Aggregator.compute_weights`, `Registry.list_runs`,
and serializes the result. If a number appears in an HTTP response, the same
number appears in the CLI output for the same run.

**P2 - Models are derived, not duplicated.** Every response shape is a
`Serializer` whose fields are a subset of an existing dataclass's `to_dict()` /
`as_dict()` output, and where the shape *is* that output the serializer derives
from it through `ModelSerializer`. Every one of those dataclasses gains a
`from_dict`, so the client can return the framework's own class instead of a
client-side copy. A field that exists only in the service layer is a bug, and a
model class that exists only on the client is a bug.

**P3 - Capability honesty.** No endpoint may claim a capability the host lacks.
`GET /sandbox/status` returns the real `CapabilityReport`, including
`usable: false` and the reasons. `POST /sandbox/exec` returns the provider's real
`ExecResult`, including exit code 127 when no guest agent is installed. There is
no `501 Not Implemented` path: either the operation exists and reports its true
outcome, or the route does not exist.

The same honesty applies to the deployment: a host without the API app does not
have a `serve` command, and its absence is a `CommandError` naming the missing
app, not an import traceback.

**P4 - No API-only state.** Run state lives in the sqlite registry
(`mactf/db.py`), not in a process dictionary. `RunManager` holds only the live
handle (thread, event bus, cancel flag). Killing the server leaves every completed
run intact and readable.

**P5 - Embedded and remote are the same program.** The client has two transports,
`InProcessTransport` and `HTTPTransport`, and one parametrized suite over both.
The in-process transport builds a real WSGI request and dispatches it through the
same `ROOT_URLCONF`, middleware chain, view, and serializer as the socket path; it
does not call the engine directly. Any behaviour that only works remotely is not
part of the client, and the two documented asymmetries (a `callable` spec, and the
concurrency semaphore) are stated where they occur.

**P6 - Scoring must never depend on the transport.** A scorecard written before
the server started is byte-identical to one written through a `POST`.
Serialization is lossless in both directions: `to_dict` -> serializer ->
`from_dict` -> `to_dict` is the identity, and a test asserts it per class.

**P7 - Fail loudly on configuration.** The existing settings validator
([mactf/conf.py:280](mactf/conf.py#L280)) raises `ImproperlyConfigured` for
contradictory settings. The API surfaces that as a 500 with the message, never as
a silent degradation.

**P8 - Additive versioning.** Everything lands under `/api/v1`. Adding a field is
a minor version; removing or retyping one requires a new prefix.

**P9 - Hierarchy is the rule, not an exception.** Peer hierarchy is a property of
the orchestrator layer the user designs. When the user composes an orchestrator
with nested peers, the framework trains, scores, aggregates, checkpoints, and
serves that hierarchy natively. A flat swarm is not a special case of the
framework; it is the degenerate tree with a single level. Hierarchy is never an
exception path, never a separate pipeline, and never a second scoring
implementation.

The mechanical consequences, all normative:

- `ScoringPipeline`, `Aggregator`, `MACTFTrainer`, `MACTFCheckpoint`, and every
  API route operate on a `PeerTree`. A flat run is `PeerTree.flat(names)`. There
  is no `if hierarchical:` branch in scoring or aggregation.
- The node labels `peer`, `subagent`, `miniagent`, `microagent`, `nanoagent` are
  display and policy labels, not a depth limit. Depth is an integer; `GENERIC`
  covers level 6 and beyond.
- No capability is withheld from a deeper level. A nano-agent can be trained,
  scored, warned, aggregated, checkpointed, and served exactly like a root peer.
- A non-offsettable violation at any depth voids every ancestor. Averaging never
  launders a T0/T1/T2 breach.

Definition in [10-hierarchical-peers.md](10-hierarchical-peers.md) section 1.

**P10 - One extension mechanism per concern.** A custom scorer is a `Scorer`
subclass, a custom objective is a `LossFunction`, a custom step is a
`TrainerProtocol`, a custom optimizer is an `OptimizerBackend`, and a custom
coordination policy is a `PeerStrategy`. There is no second, informal way to do
any of those, and no framework-internal branch that bypasses the protocol for
the built-in case.

**P11 - Two numbers, two questions, no reconciliation.** `final_score` is the
unbounded training signal the docx equations define. `final_points` is the bounded
0-100 rubric score the framework's own text implies. They are computed from the
same ledger by independent paths, they are allowed to disagree, and neither may
be derived from the other. `final_score` is never rescaled onto 0-100, and
`final_points` never feeds `compliance.constrain`.

The reason this is a principle rather than an implementation detail: the two
quantities have opposite stability requirements. `final_score` must be bit-stable
across a resume and across a version upgrade, because an optimizer mid-run is
comparing episodes. `final_points` must be able to change when the framework's
rubric interpretation changes, because it is what a human reads and what a
leaderboard ranks. A single number cannot satisfy both, and every attempt to
fuse them ends with one requirement silently failing.

The mechanical consequences:

- `continuous.py` does not import `points.py`, and `points.py` does not import
  `continuous.py`. A test enforces both directions.
- `SCORING.POINTS` and `SEVERITY.WEIGHTS` are separate tables with separate
  validation. They are not merged, and a shared "severity" helper that returns
  one table's fraction is a bug.
- Guards deny points and do not deny `final_score`. A reward event present in the
  ledger contributes to `R` whether or not its guard passes.
- `PeerScorecard` gains fields rather than changing them. Nothing that was
  serialized before this document has a new meaning.
- The aggregation input is explicitly selectable
  (`COOPETITION.SIGNAL_SOURCE`), so an operator can run either economy without
  forking the library.

Definition in [11-points-and-normalization.md](11-points-and-normalization.md)
sections 2, 6, and 10.

**P12 - The service layer is an app, not a package.** `mactf/http/` is the
transport: `Request`, `Response`, URL conf, middleware, and a WSGI handler. It
knows nothing about scoring. `mactf/contrib/api/` is the service: an `AppConfig`
that supplies `urls.py`, `views.py`, `serializers.py`, `auth.py`, `checks.py`, and
`management/commands/serve.py`, selected by `INSTALLED_APPS` exactly as
`django.contrib.admin` is. Removing the app removes the API, its routes, its
system checks, and its `serve` command in one settings edit, with no import error
and no effect on `mactf run`.

Three consequences follow, and each is testable:

- **An installed app can contribute a URL conf.** `AppConfig.get_urls()` returns
  a module or a pattern list, `ROOT_URLCONF` includes it with `include()`, and
  `reverse()` names come from that app. A second app (`mactf.contrib.scoring`)
  can contribute routes the same way.
- **An installed app can contribute commands and migrations.** `serve` lives in
  `mactf/contrib/api/management/commands/`, discovered from
  `INSTALLED_APPS`, not hardcoded in core. Its tables migrate under the app's own
  label.
- **An installed app can contribute checks.** `mactf.api.E001` and friends are
  registered with `register(check, app="api")` and run in `mactf check` only when
  the app is installed.

The corollary is the client. Because the in-process transport dispatches through
`ROOT_URLCONF`, the client is not an alternative to the server; it is the server,
called without a socket. That is why `mactf.client` can live in core while the
service it talks to lives in an opt-in app: the client depends on the transport
layer, and the app depends on nothing the client provides.

Definition in [12-http-core.md](12-http-core.md) and
[13-app-integration.md](13-app-integration.md).

## 5. Non-goals

- **Not a training-correctness rewrite.** Phase A closes the six capability gaps
  listed above and phase A0 adds the wrapper seams; neither re-derives the
  scoring mathematics, and neither touches `docs/framework.md`. Every A0
  refactor must reproduce today's numbers to the tolerance its equivalence gate
  names. The one additive score (the 0-100 rubric, gap 16) is deliberately a
  second quantity: it is computed beside `final_score` and cannot alter it.
- **Not a replacement of the framework's equations with a rubric.** The points
  table in [11](11-points-and-normalization.md) is a projection for reporting,
  ranking, and calibration. It is not the training objective, and no optimizer
  step reads it unless an operator sets `COOPETITION.SIGNAL_SOURCE="points"`,
  which is documented with its own kappa rescaling and its own equivalence
  tests.
- **Not a multi-tenant SaaS.** One process serves one settings object and one
  project directory. Isolation between tenants is out of scope; run-level
  isolation within a process is in scope.
- **Not a model-serving gateway.** MACTF trains swarms; it does not host inference
  endpoints for the trained peers.
- **Not a replacement for the CLI.** Every capability added here must remain
  reachable from `manage.py`, and `mactf serve` is a management command like any
  other.
- **No WebSocket.** Core ships no WebSocket server, no ASGI-only route, and no
  channel layer. Progress is streamed as server-sent events over an ordinary WSGI
  generator response, which works on the standard library and behind every reverse
  proxy an operator is likely to have. A WebSocket transport, if anyone wants one,
  is a separate package in the shape of Django Channels, not a core concern.
- **No new mandatory dependencies.** A bare `pip install mactf` keeps exactly the
  current dependency set (`torch>=2.4`, `PyYAML>=6.0`) and gains a working server
  (`wsgiref`) and client (`urllib.request`) from the standard library. `scipy`,
  `peft`, `transformers`, `bitsandbytes`, `tensorboard`, and `wandb` remain
  extras. `uvicorn` is an optional `[asgi]` extra for operators who want the ASGI
  adapter; nothing in the framework requires it, and the `stdlib-only` CI job
  proves that by installing the package with no third-party HTTP library present.
- **Not a reimplementation of Hugging Face `Trainer`.** `MACTFTrainer` accepts
  any `nn.Module`, including a `PreTrainedModel`, and drives it through the
  framework's scoring and aggregation. It does not reimplement the HF training
  loop, its callbacks, its distributed launch, or its checkpoint format.
- **Not a distributed optimizer.** `MACTFOptimizer` operates on one process's
  parameters. Cross-host gradient sharding is document 05's scope, and even there
  the optimizer is local.
- **Not an inference server.** `POST /runs/{id}/infer` runs inference on a loaded
  run's tree. It is not a general model-serving gateway with batching queues,
  autoscaling, or SLA guarantees.
- **No automatic topology design.** The framework does not decide how many
  sub-agents to spawn. `HierarchySpec` describes a topology the user chose;
  `fanout_policy="random"` builds a given shape randomly and seeded.

## 6. Compatibility policy

| Change | Allowed in | Mechanism |
|---|---|---|
| New endpoint | minor | additive under `API.BASE_PATH` |
| New optional field in a response | minor | serializer field with a default |
| New optional field in a request | minor | serializer field with a default |
| New required field in a request | major | new path prefix |
| Removing or renaming a field | major | new path prefix |
| Changing an enum value | major | new path prefix |
| Changing a status code for the same condition | major | documented in the contract |
| New scope | minor | `mactf/contrib/api/auth.py` constant + docs |
| New client method | minor | new `Transport` method is a major change; add a free function instead |
| Adding a server route without a `name=` | major | `reverse()` breaks; criterion 46 |

Server semver is the package semver. `mactf.__version__` is the single source of
truth ([mactf/__init__.py:15](mactf/__init__.py#L15)) and stays so. Installing or
removing `mactf.contrib.api` changes no version, exactly as toggling
`django.contrib.admin` does not change Django's.

## 7. What "real, not a stub" means concretely

This plan is accepted only if each of the following is true at the end of its
phase. These are the acceptance criteria the implementation is measured against.

| # | Criterion | Where verified |
|---|---|---|
| 1 | `POST /runs` starts a real training run whose every epoch is visible in `GET /runs/{id}/episodes`, and the scorecards match `manage.py run --json` for the same seed | `tests/contrib_api/test_runs.py`, `tests/e2e/test_api_cli_parity.py` |
| 2 | `POST /evaluations` with a hand-written P1 event returns a scorecard whose `floor` is negative and whose `is_void` is true, with no reward applied | `tests/contrib_api/test_evaluations.py` |
| 3 | Every rule ID in the catalog can be triggered through `POST /evaluations` by supplying evidence, and the resulting severity matches the rule's default | `tests/evaluator/test_rule_reachability.py` |
| 4 | A warning issued in run A is still active in run B after `--resume`, without a checkpoint file, when a checkpoint is unavailable | `tests/contrib_api/test_resume.py` |
| 5 | `InProcessTransport` and `HTTPTransport` produce identical framework objects (`Run`, `PeerScorecard`, `AggregationReport`) for the same request, and neither path returns a class the other cannot | `tests/client/test_parity.py` |
| 6 | `GET /sandbox/status` on a host without QEMU returns `usable: false` with reasons, and `mactf run` still works via `LocalSandbox` | `tests/contrib_api/test_sandbox.py` |
| 7 | `POST /sandbox/exec` with a provisioned image and the guest agent returns the command's real stdout and exit code | `tests/sandbox/test_guest_agent.py` |
| 8 | `bootstrap` with `BOOTSTRAP=True` produces a measurable loss decrease over the priming data and updates both peers and the reference | `tests/test_bootstrap.py` |
| 9 | `GET /runs/{id}/events` replays from `Last-Event-ID` with no gaps and no duplicates | `tests/contrib_api/test_events.py` |
| 10 | `POST /runs/{id}/cancel` stops within one epoch boundary and the registry row ends in `cancelled` | `tests/contrib_api/test_cancel.py` |
| 11 | Clicking through the OpenAPI schema at `/docs` lists every endpoint in document 02 and no endpoint that is not in it | `tests/contrib_api/test_contract_completeness.py` |
| 12 | Two in-process workers complete an epoch distributed and the aggregate equals the single-host aggregate for the same inputs | `tests/distributed/test_two_workers.py` |
| 13 | `ScoringPipeline` with the three default scorers produces `final_score` equal to today's `Scorer` for every fixture ledger, to `1e-12` | `tests/scoring/test_pipeline_equivalence.py` |
| 14 | `RewardScorer`, `PenaltyScorer`, and `WarningScorer` each equal their `continuous.py` counterpart, including Omega and the warning-only exclusion | `tests/scoring/test_{reward,penalty,warning}_scorer.py` |
| 15 | A warning that matures into a penalty in the same episode is counted exactly once, and an inverted scorer order raises at construction | `tests/scoring/test_pipeline_validation.py` |
| 16 | A custom `RewardScorer` cannot bypass the floor: a ledger with P1 still yields `final_score < 0` and `is_void` true | `tests/scoring/test_pipeline_equivalence.py` |
| 17 | `CompositeLoss` and `GAEAdvantage` reproduce `compute_loss` and `generalized_advantage` to `1e-12` | `tests/rl/test_loss_functions.py`, `tests/rl/test_advantage_functions.py` |
| 18 | `MACTFTrainer()` with defaults reproduces `Orchestrator.run()` for the same seed to `1e-9` on every score, kappa, and entropy | `tests/training/test_trainer_equivalence.py` |
| 19 | A PEFT/LoRA fine-tune trains only adapter parameters, and a `ParamGroupRouter` that leaves any trainable parameter unclaimed raises with the parameter name | `tests/training/test_finetune.py`, `tests/training/test_optimizer_router.py` |
| 20 | `ScipyOptimizer` with `L-BFGS-B` reduces a quadratic loss monotonically and does not accumulate gradients across evaluations | `tests/training/test_optimizer_scipy.py` |
| 21 | A `weight_update` that is `None`, raises, or reports `fallback` results in `KappaSoftmaxAverage`, with the reason present in `AggregationReport.adjustments` | `tests/training/test_weight_update.py` |
| 22 | `MACTFLogger` writes one JSONL record per event, `tail()` returns them in order, and a raising sink does not fail a training step | `tests/logging/test_logger_async.py`, `tests/logging/test_logger_failure.py` |
| 23 | Forcing a NaN loss causes `MACTFWatchdog` to snapshot before it aborts, and the snapshot file verifies | `tests/watchdog/test_actions.py` |
| 24 | Killing a run at epoch 3 and calling `resume_from_last_checkpoint()` makes epochs 4-6 bit-identical to the uninterrupted run under `DETERMINISTIC=True` | `tests/checkpoints/test_resume_bit_identical.py` |
| 25 | A truncated newest checkpoint is skipped by `latest(valid_only=True)` and the skip appears in `ResumeReport.skipped_checkpoints` | `tests/checkpoints/test_latest_valid.py` |
| 26 | A depth-1 `PeerTree` reproduces today's run output exactly, and its `root_paths` equal today's `Episode.participants` keys | `tests/hierarchy/test_flat_equivalence.py` |
| 27 | A non-offsettable P1 at `peer-a/sub-1/mini-2/micro-1/nano-1` voids `peer-a` and every node above it, regardless of rewards earned by siblings | `tests/hierarchy/test_branch_void.py` |
| 28 | A three-level tree trains, scores, and aggregates level by level; a heterogeneous branch is legal while siblings with mismatched layouts are rejected | `tests/hierarchy/test_train_epoch.py`, `tests/hierarchy/test_aggregation_groups.py` |
| 29 | `run_inference` on the same tree produces identical outputs for `parallel=True` and `parallel=False`, and identical outputs across two `PYTHONHASHSEED` values | `tests/hierarchy/test_inference.py`, `tests/hierarchy/test_inference_determinism.py` |
| 30 | Inference over a budget raises `ContractError` naming the offending path and the remaining budget, rather than returning a partial result | `tests/hierarchy/test_inference.py` |
| 31 | The 17 reward weights sum to exactly `100.0`, every one of the 40 catalog rules has exactly one points binding, and no binding names a rule that is not in the catalog | `tests/scoring/test_points_table.py` |
| 32 | Appending any reward event never lowers `final_points`; appending any penalty event never raises it; raising a severity never lowers a penalty's magnitude | `tests/scoring/test_points_monotone.py` |
| 33 | An episode that earned rewards but also fired a non-offsettable penalty has `final_points == 0.0` and `void is True`, while its `gross_points` remains the earned value | `tests/scoring/test_points_monotone.py` |
| 34 | Every guard denial appears in `PointsBreakdown.denied` with a reason and the missing evidence fields, and no denial is silently dropped | `tests/scoring/test_guards.py` |
| 35 | Running the pipeline with the points stages enabled and disabled produces identical `final_score`, `floor`, `is_void`, and `continuous` for every fixture ledger | `tests/scoring/test_points_equivalence.py` |
| 36 | A four-peer cohort of `20, 40, 60, 80` calibrates to `51.5, 58.5, 65.5, 72.5` at the defaults, order is preserved under permutation, and void peers are pinned to `0.0` and excluded from the statistics | `tests/scoring/test_normalize.py`, `tests/scoring/test_normalize_invariants.py` |
| 37 | `GET /scores/scale` and `client.scores.scale()` return the same 40 bindings, severity fractions, and `table_version`, and `PointsTable.from_dict` reconstructs the framework object | `tests/contrib_api/test_points_routes.py`, `tests/client/test_points_parity.py` |
| 38 | A ledger with a reward and a P1 yields `final_score > 0` and `final_points == 0.0` simultaneously, on both transports, and neither value is derived from the other | `tests/scoring/test_points_equivalence.py`, `tests/client/test_points_parity.py` |

Criteria 39-54 cover the integration this revision adds. They are stated
normatively in [12-http-core.md](12-http-core.md) section 16 (39-46, the
transport) and [13-app-integration.md](13-app-integration.md) section 10 (47-54,
the app and client), and restated here so this table stays the single index.

| # | Criterion | Where verified |
|---|---|---|
| 39 | `import mactf.http` succeeds in an interpreter with no third-party HTTP package installed | `tests/http/test_stdlib_only.py` |
| 40 | A route declared with `path(..., name="x")` can be produced by `reverse("x", ...)` and consumed by `resolve()`, for all five converters | `tests/http/test_urls.py` |
| 41 | `get_wsgi_application()` is callable by `wsgiref.simple_server` and by `mactf.testing.Client` without modification | `tests/http/test_handler.py` |
| 42 | An unhandled exception in a view produces a `500` problem body with a trace id, and no traceback in the body unless `DEBUG` | `tests/http/test_middleware.py` |
| 43 | Removing `mactf.contrib.api` from `INSTALLED_APPS` leaves every module in `mactf/http` importable and every test in `tests/http/` passing | `tests/http/test_layering.py` |
| 44 | No module under `mactf/http/` imports an engine module, verified by parsing imports | `tests/http/test_layering.py` |
| 45 | `mactf.http.openapi.schema()` emits an operation for every route in `ROOT_URLCONF` with `operationId == route name` | `tests/http/test_openapi.py` |
| 46 | SSE streams through `wsgiref` with heartbeats, and a closed connection releases the subscription without leaking a thread | `tests/http/test_streaming.py` |
| 47 | `INSTALLED_APPS` without `mactf.contrib.api` leaves `mactf run`, `mactf check`, `mactf migrate`, and `MACTFClient()` working, and `mactf serve` absent with a helpful error | `tests/contrib_api/test_uninstalled.py` |
| 48 | `mactf serve` is discovered through `command_packages()` from the app, not from a core module | `tests/contrib_api/test_serve.py` |
| 49 | A project's `ROOT_URLCONF`, `MIDDLEWARE`, `APPEND_SLASH`, `HTTP.*`, and `API.*` overrides take effect, proving they are in `DEFAULTS` | `tests/contrib_api/test_settings_overrides.py` |
| 50 | `mactf.contrib.api` registers `mactf.api.E001`, `W002`, `E003`, and `E004`, and `mactf check` reports them | `tests/contrib_api/test_checks.py` |
| 51 | An app declaring `migrations_module` has its statements applied after core's, recorded in `mactf_migrations` | `tests/contrib_api/test_migrations.py` |
| 52 | `from_dict(to_dict(x)) == x` for every class in doc 13 section 5, and `from_dict` ignores an unknown key | `tests/client/test_from_dict.py` |
| 53 | `InProcessTransport` and `HTTPTransport` return identical framework objects for the same request, including identical exception classes for a `422`, `404`, `409`, and `502` | `tests/client/test_parity.py` |
| 54 | `mactf.testing.Client` serves a request with no socket, and its response bytes equal those of `InProcessTransport` | `tests/testing/test_client.py` |

## 8. Risks and their mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Torch holds the GIL during large tensor ops | The API starves under concurrent runs | Runs execute in their own thread; all torch work happens there. Read endpoints touch sqlite only. `wsgiref` serves one request per thread and `RunManager` caps concurrent runs (`MAX_CONCURRENT_RUNS`) |
| A run thread outlives the process | Orphaned threads, half-written checkpoints | `atexit` shutdown hook marks `interrupted` runs; checkpoints are already atomic (`atomic_torch_save`) |
| Evaluator ingest becomes a scoring bypass | Callers could fabricate rewards | Ingest validates every event through `Ledger.record`, which rejects unknown rules and negative magnitudes; rewards are additionally gated by the non-offsettable check in `compliance.constrain` |
| Distributed aggregation drifts from single-host | Two different notions of "the score" | One code path: the coordinator calls the same `Aggregator`; workers never compute weights |
| `docs/plans` rots against the code | The documents become misleading | `tests/contrib_api/test_contract_completeness.py` parses the endpoint table in document 02 and fails when the app's route table disagrees |
| A client model family reappears beside the framework's dataclasses | Two shapes drift and a dashboard reads a stale field as `None` | `mactf/client/` imports its return types from `mactf`; `tests/client/test_no_parallel_models.py` walks the package and fails on any locally defined dataclass whose name matches an engine class. The four client-only envelopes (`Run`, `Episode`, `RuleEvent`, `RunHandle`) are listed in doc 03 section 6.1 so the exemption is a decision, not an accident |
| In-process dispatch skips a serializer and hides a wire bug | Tests pass locally and the socket path returns a malformed body | `InProcessTransport` builds a WSGI environ and calls `get_wsgi_application()`; it never calls a view directly. Criterion 53 asserts byte-identical `JSON` from both transports for every endpoint |
| `ROOT_URLCONF` becomes a second source of truth for routes | A route exists in the app but is unreachable, or vice versa | The app's `urls.py` is the only route table; `reverse()` is tested per name (criterion 41) and the OpenAPI document is generated from the same conf (46) |
| The stdlib transport grows into a web framework | Maintenance cost with no benefit, and a second Django | `mactf/http/` is capped at the surfaces listed in doc 12: seven modules, no ORM, no forms, no template engine, no admin. Layering test 39 and the "transport, not framework" rule in 12 section 1 enforce the boundary |
| The scorer split silently changes a number | Every historical run becomes incomparable | The numerical-equivalence gate (07 section 11) runs against the existing `tests/test_continuous.py` and `tests/test_ledger.py` fixtures at `1e-12` before `hierarchy.py` starts |
| The wrapper layer becomes a second implementation | Two code paths that drift | `MACTFTrainer` delegates to `Aggregator`, `MACTFOptimizer` delegates to `torch.optim`, `KappaSoftmaxAverage` delegates to `compute_weights`. A reimplementation in a wrapper is a review rejection |
| Hierarchy turns a small breach into an unnoticed one | A nano-agent's P1 averaged away by well-behaved siblings | `branch_void` is computed bottom-up and passed to `compliance.constrain` at every ancestor; acceptance criterion 27 tests the depth-5 case |
| A resume that is really a warm restart is presented as a resume | Silent divergence from the uninterrupted run | `ResumeReport.bit_identical_expected` is true only when RNG, data, memory, and optimizer sections were all restored, and the note says so |
| A PEFT adapter silently never trains | A fine-tune that reports success and learned nothing | `MACTFOptimizer` asserts every trainable parameter is claimed exactly once by a router route or default, and names the unclaimed parameter |
| Per-node seeds depend on `PYTHONHASHSEED` | A hierarchical run is not reproducible across processes | Per-node seeds derive from `blake2b`, and acceptance criterion 29 runs the same tree under two `PYTHONHASHSEED` values |
| The points table drifts into three copies | A scorecard says 47, the client says 45, and the docs say 50 | One `DEFAULT` in `points.py`, serialized by `GET /scores/scale`, asserted equal to `PointsTable.from_dict` by criterion 37. No route, docstring, or test may hardcode a weight |
| A calibration bug flattens the training signal | Every peer gets the same weight and the run stops learning | `shrink` defaults to `0.35`, not `0.0`; invariant N2 asserts order preservation; criterion 36 asserts a 60-point spread becomes 21 rather than 0 |
| Someone "fixes" the `final_score`/`final_points` divergence | Every historical scorecard becomes incomparable, or the rubric silently becomes the training signal | P11 in section 4 states the divergence is normative; criterion 38 asserts the exact case that looks like a bug; the equivalence gate fails if `final_score` moves |
| Points are computed before compliance and a bug suppresses a scorecard | A scoring regression turns into a missing run | Step `4a` in 07 section 7 runs after `ContinuousScore` and `compliance.constrain`, so a points failure cannot prevent a scorecard being written |

## 9. Open items requiring the user

1. **Two Gemini threads.** `7ac3731679b9b5fd` and `8f7dcbb2ed0643e0` are not
   readable without a session. If they contain interface requirements, export
   them into `docs/` and they will be folded into documents 02-06 before Phase B.
2. **Authentication model.** Document 02 specifies static bearer tokens with
   scopes, which is sufficient for a single-team deployment. If MACTF must run
   multi-tenant or behind an identity provider, that choice changes
   `mactf/contrib/api/auth.py` and the scope table, and should be made before
   Phase B.
3. **Persistence backend.** The plan keeps sqlite. A Postgres backend would
   change `mactf/db.py` behind the same `Registry` interface; the plan isolates
   that interface deliberately, but does not implement it.
4. **`Scorer` shim removal.** `mactf.scoring.pipeline.Scorer` continues to mean
   `ScoringPipeline` with a `DeprecationWarning` through `0.2.x` and is removed
   in `0.3.0`. If a downstream user needs a longer window, that is a scheduling
   decision the plan assumes is not needed.
5. **Checkpoint schema versions.** `MACTFCheckpoint` writes `schema_version=2`
   and reads version 1 through a documented migration. Keeping the v1 reader
   indefinitely is a maintenance cost; the plan assumes it is dropped when no
   v1 checkpoint predates the release.
