---
title: MACTF Service and Client Plans
status: approved-for-implementation
version: 1.2.0
applies_to: mactf >= 0.1.0
owner: TBD
last_reviewed: 2026-10-21
---

# MACTF Service and Client Plans

This directory is the build specification for the service layer and the Python
client of the Multi-Agent Coopetition Training Framework. It is normative for
implementation: where a document here and a document in `docs/` disagree,
`docs/framework.md` controls scoring semantics and this directory controls
interface shape.

The service layer is an **installed app** (`mactf.contrib.api`, selected by
`INSTALLED_APPS`) mounted through `ROOT_URLCONF`, on a **standard-library
transport** (`mactf/http/`), with the **client in core** (`mactf.client`, two
transports, one set of objects). That arrangement is the whole plan in one
sentence, and everything below follows from it.

Nothing here is a sketch. Every endpoint, serializer, module path, and invariant
below resolves to a concrete class that either exists in `mactf/` today or has its
file and signature fixed by the plan. There are no placeholder endpoints, no
"return a canned response", and no capability the host cannot actually deliver.

## Reading order

| Order | Document | Purpose |
|---|---|---|
| 1 | [00-scope-and-audit.md](00-scope-and-audit.md) | Goals, non-goals, and the eighteen verified gaps in the current library |
| 2 | [01-target-architecture.md](01-target-architecture.md) | Module graph, layering, concurrency model, lifecycle |
| 3 | [12-http-core.md](12-http-core.md) | The stdlib transport: `Request`, `Response`, URL conf, middleware, WSGI, serializers |
| 4 | [13-app-integration.md](13-app-integration.md) | `AppConfig` extensions, `mactf.contrib.api`, `mactf.client`, `mactf.testing.Client` |
| 5 | [02-api-contract.md](02-api-contract.md) | Complete HTTP/SSE contract, serializers, errors, auth, route names |
| 6 | [03-python-sdk.md](03-python-sdk.md) | The client: one protocol, two transports, identical framework objects |
| 7 | [04-gap-closures.md](04-gap-closures.md) | The twelve internal implementations that make the API substantive |
| 8 | [05-distributed-runs.md](05-distributed-runs.md) | Multi-host coordinator and worker protocol as an app |
| 9 | [06-verification-and-release.md](06-verification-and-release.md) | Test matrix, acceptance criteria, packaging, rollout gates |
| 10 | [07-scorers-and-functions.md](07-scorers-and-functions.md) | `Scorer` ABC with `Reward`/`Penalty`/`Warning` children, the `Value`/`Advantage`/`Loss` wrappers, and the points projection |
| 11 | [10-hierarchical-peers.md](10-hierarchical-peers.md) | `PeerTree`, arbitrary-depth sub-peers, hierarchical scoring rollup and inference |
| 12 | [08-trainer-optimizer-strategies.md](08-trainer-optimizer-strategies.md) | `MACTFTrainer`, `MACTFOptimizer`, PEFT/LoRA, peer strategies and weight updates |
| 13 | [09-logging-watchdog-checkpoint.md](09-logging-watchdog-checkpoint.md) | `MACTFLogger`, `MACTFWatchdog`, `MACTFCheckpoint` and resumable training |
| 14 | [11-points-and-normalization.md](11-points-and-normalization.md) | The 0-100 rubric, the guard catalogue, and cohort calibration |

Documents 12 and 13 are prerequisite reading for 02 and 03, because they define
the words those documents use: a `Serializer` is not a Pydantic model, a route is
a `name=`d `path()` in an app's `urlpatterns`, and a "backend" is a transport
over a WSGI call. Reading 02 or 03 first is possible but produces the wrong mental
model of where the code lives.

Documents [04](04-gap-closures.md), [05](05-distributed-runs.md), and
[06](06-verification-and-release.md) are subordinate to 02 and 03: the contract
and the client protocol define what must exist, and the gap closures define how
the underlying library becomes capable of serving it honestly.

Documents 10-14 are the capability layer. Their **reading** order is 07, 10, 08,
09 because the hierarchy types change the trainer and checkpoint signatures.
Their **document numbers** keep the plan's original numbering so no earlier
cross-reference breaks: 07, 08, 09, 10.

Document 11 (points) is last because it depends on 07 for the scorer stages and on
10 for `branch_void`, and because it is the only document that is purely additive
to the numbers: everything it specifies can be deleted without changing
`final_score`. Read it after 07 even if you never intend to use points, since 07
section 4.1 introduces the methods it defines.

## Status table

| Deliverable | Depends on | Blocks | Phase |
|---|---|---|---|
| `mactf/data/` task loading (04.1) | - | Anything dataset-driven | A |
| `mactf/orchestration/bootstrap.py` (04.2) | 04.1 | Real bootstrap in `phases.py` | A |
| `mactf/scoring/evaluator.py` (04.3) | - | Every catalog rule reachable from evidence | A |
| `mactf/db.py` migration 2 (04.4) | 04.3 | Warning resume, adjudication history | A |
| `mactf/sandbox/guest_agent.py` (04.5) | - | `QemuSandbox.exec` honesty | A |
| `mactf/coopetition/pool.py` (04.6) | 04.3 | framework.md 9.2 reporting | A |
| `mactf/scoring/{base,reward_scorer,penalty,warning_scorer}.py` (07) | - | The scorer split | A0 |
| `mactf/rl/functions/` (07) | 07 scorers | `MACTFTrainer` objectives | A0 |
| `mactf/coopetition/{tree,delegation,synthesis}.py` (10) | - | Hierarchical runs and inference | A0 |
| `mactf/scoring/hierarchy.py` (10) | 07 scorers, 10 tree | Hierarchical rollup and `branch_void` | A0 |
| `mactf/training/{optimizer,trainer,finetune,protocol}.py` (08) | 07, 10 | `MACTFTrainer` entry point | A0 |
| `mactf/coopetition/{strategy,weights}.py` (08) | 08 trainer | Peer strategies and kappa fallback | A0 |
| `mactf/training/logging.py`, `watchdog.py`, `orchestration/checkpoint_v2.py` (09) | 08 | Observable and resumable runs | A0 |
| `mactf/scoring/{points,evidence,guards,normalize}.py` (11) | 07, 10 | The 0-100 rubric and cohort calibration | A0 |
| `mactf/db.py` migration 3 (11) | 11 points | Points and normalization persistence | A0 |
| `mactf/apps.py` extensions, app-scoped commands and migrations (13) | - | The service app, `mactf serve` | B |
| `mactf/http/` (12) | 13 app extensions | The service app, the client | B |
| `mactf/orchestration/live.py` (01.3) | - | Shared run manager for the app and the client | B |
| `mactf/contrib/api/` (02, 13) | Phase A, 12, 13 | `mactf serve`, the `/api/v1` surface | B |
| `mactf.testing.Client` (13) | 12, 13 | Server-free integration tests | B |
| `from_dict` on nine framework classes (13) | - | Any client return value | B+C |
| `mactf/client/` (03) | 12, 13 | `mactf.client.MACTFClient` | C |
| `mactf/distributed/` (05) | 02, 03 | Multi-host training | D |

Phase A is library-only and ships independently, split into **A** (the six gap
closures) and **A0** (the wrapper refactor). A0 depends on nothing in A and adds
no new mandatory dependency. Phases B, C, and D each end with the full test suite
green and `twine check` passing.

Nothing here is gated on an extra. `pip install mactf` gives you `mactf serve`
(standard library `wsgiref`) and `mactf.client.MACTFClient` (standard library
`urllib.request`). The single optional extra this plan adds is `asgi`, for an
operator who wants the ASGI adapter in front of `uvicorn`.


## Glossary of terms used in these documents

These are the framework's own terms, not newly invented ones. Each is defined in
`docs/framework.md`; the short form here exists so the API documents can be read
without constant cross-referencing.

| Term | Meaning | Source |
|---|---|---|
| Tier T0-T6 | Precedence ladder from safety/legality down to efficiency/style | framework.md 3 |
| Severity S0-S4 | Impact tier, from none to severe/irreversible | framework.md 4.1 |
| R1-R17 | Reward rules | framework.md 5 |
| P1-P23 | Penalty rules | framework.md 6 |
| W1-W9 | Warning rules (parameters, not rules) | framework.md 8 |
| Non-offsettable | P1, P2, P17, P20, P21, P22 set an absolute floor | framework.md 6.1 |
| Floor | `< 0` value no reward in the same episode may reduce | `mactf/compliance.py` |
| Omega | Bounded amplifier applied to P20/P21/P22 | framework.md 3.1, docx Turn 3 |
| kappa | Adaptive softmax temperature over peer scores | docx Turn 5 |
| Weight cap/floor | Per-peer aggregation bounds enforced by simplex projection | framework.md 10.1.2-3 |
| Void episode | Episode whose work earns no task credit due to a T0/T1/T2 breach | framework.md 3.1.2 |
| Competitive component | Zero-sum peer ranking | framework.md 9.2 |
| Cooperative component | Non-zero-sum additive pool (R10, R11, R16) | framework.md 9.2 |
| `Scorer` | Abstract base for one scoring stage | [07](07-scorers-and-functions.md) |
| `ScoreContext` | Frozen input handed to every scorer stage | [07](07-scorers-and-functions.md) |
| `ScorerOutput` | One stage's amount, breakdown, rules, and mutations | [07](07-scorers-and-functions.md) |
| `ScoringPipeline` | Concrete composite: warning, reward, penalty, then compliance | [07](07-scorers-and-functions.md) |
| `branch_void` | Non-offsettable rule ids inherited from a node's descendants | [07](07-scorers-and-functions.md), [10](10-hierarchical-peers.md) |
| `ValueFunction` / `AdvantageFunction` / `LossFunction` | Protocol wrappers over the `mactf/rl` free functions | [07](07-scorers-and-functions.md) |
| `TrainerProtocol` | `setup` / `train_step` / `on_epoch_end` custom training step | [08](08-trainer-optimizer-strategies.md) |
| `MACTFOptimizer` | Facade over torch, scipy, callable, and routed optimizers | [08](08-trainer-optimizer-strategies.md) |
| `MACTFTrainer` | Fan-out wrapper driving N peer copies or a tree | [08](08-trainer-optimizer-strategies.md) |
| `PeerStrategy` | Coordination policy across peers or levels | [08](08-trainer-optimizer-strategies.md) |
| `WeightUpdateStrategy` | How parameter vectors are blended | [08](08-trainer-optimizer-strategies.md) |
| `KappaSoftmaxAverage` | Default and fallback: kappa-controlled softmax of peer scores | [08](08-trainer-optimizer-strategies.md) |
| `PeerNode` / `PeerTree` | One node in the hierarchy and the whole topology | [10](10-hierarchical-peers.md) |
| `HierarchySpec` | Declarative topology the orchestrator asks for | [10](10-hierarchical-peers.md) |
| `DelegationPolicy` | Which children a parent sends the task to | [10](10-hierarchical-peers.md) |
| `SynthesisFunction` | How a parent forms a proposal from its children | [10](10-hierarchical-peers.md) |
| `MACTFLogger` | Typed, multi-sink run logger wired to signals | [09](09-logging-watchdog-checkpoint.md) |
| `MACTFWatchdog` | Health probes and snapshot/rollback/prune/abort actions | [09](09-logging-watchdog-checkpoint.md) |
| `MACTFCheckpoint` | Complete checkpoint payload with manifest and resume | [09](09-logging-watchdog-checkpoint.md) |
| `HealthReport` / `ProbeResult` | Watchdog verdict and one probe's evidence | [09](09-logging-watchdog-checkpoint.md) |
| `ResumeReport` | What a resume actually restored, and what it could not | [09](09-logging-watchdog-checkpoint.md) |
| `CheckpointPayload` | The versioned set of checkpoint sections | [09](09-logging-watchdog-checkpoint.md) |
| `final_score` | Unbounded docx training signal; stable across resumes and upgrades | framework.md, `mactf/compliance.py` |
| `final_points` | Bounded 0-100 rubric score; may legitimately disagree with `final_score` | [11](11-points-and-normalization.md) |
| `PointsTable` | The 40 rule bindings, severity fractions, and warning notional | [11](11-points-and-normalization.md) |
| `PointsBreakdown` | One peer-episode's reward, penalty, gross, and pinned `final_points` | [11](11-points-and-normalization.md) |
| `EpisodeEvidence` / `ClaimAudit` | Structured facts a guard reads, and the claim scan it reads | [11](11-points-and-normalization.md) |
| `GuardVerdict` | `earned` / denied / `missing` for one rule, with reason | [11](11-points-and-normalization.md) |
| `DenialRecord` | Why a rule earned nothing, including the missing evidence fields | [11](11-points-and-normalization.md) |
| `NormalizationPolicy` | `target_mean`, `shrink`, void exclusion, signal scale | [11](11-points-and-normalization.md) |
| `CohortStats` / `CohortNormalizer` | One episode's points distribution and the transform over it | [11](11-points-and-normalization.md) |
| `calibrated` | Cohort-relative display score, order-preserving, void-pinned | [11](11-points-and-normalization.md) |
| `SIGNAL_SOURCE` | `"points"` or `"final_score"`; which economy feeds the aggregator | [11](11-points-and-normalization.md), [08](08-trainer-optimizer-strategies.md) |
| `Request` / `Response` | The stdlib HTTP objects; `mactf.http`'s whole surface | [12](12-http-core.md) |
| `path` / `include` / `reverse` / `resolve` | URL conf primitives; a route is a `name=`d `path()` | [12](12-http-core.md) |
| `View` / `as_view` | Class-based view and its constructor; a view returns a `Response` | [12](12-http-core.md) |
| `get_wsgi_application()` | The one WSGI entry point; `mactf serve` and every test client call it | [12](12-http-core.md) |
| `Serializer` / `Field` | Declarative wire model over an existing `to_dict`/`from_dict` pair | [12](12-http-core.md) |
| `ROOT_URLCONF` / `MIDDLEWARE` | Settings that mount the app and build the request chain | [13](13-app-integration.md) |
| `mactf.contrib.api` | The service app: `urlpatterns`, views, checks, and the `serve` command | [13](13-app-integration.md) |
| `AppConfig.get_urls()` / `command_packages()` | How an installed app contributes routes and commands | [13](13-app-integration.md) |
| `mactf_migrations` | App-scoped migration table keyed by `(app, version)` | [13](13-app-integration.md) |
| `Transport` | `InProcessTransport` or `HTTPTransport`; the client's only seam | [03](03-python-sdk.md) |
| `InProcessTransport` | Dispatches through WSGI and `ROOT_URLCONF` with no socket | [03](03-python-sdk.md) |
| `MACTFClient` | The client; namespaces `runs`, `scores`, `scoring`, `rules`, `peers`, `sandbox`, `trainers` | [03](03-python-sdk.md) |
| `mactf.testing.Client` | The in-process test client; replaces `fastapi.testclient.TestClient` | [13](13-app-integration.md) |

## What this plan deliberately does not do

- It does not change `docs/framework.md` or `docs/framework.md.bak`. Both are
  read-only normative inputs.
- It does not rescale, replace, or reinterpret `final_score`. The docx equations
  keep producing the number they produce today; the rubric is a second, additive
  quantity and the two are documented as legitimately divergent.
- It does not make scoring depend on a network service. The transport is a thin
  layer over the same objects the CLI uses; killing the server never invalidates a
  scorecard already written to the registry, and `InProcessTransport` answers the
  same questions with no server at all.
- It does not introduce a second source of truth for any model. Every wire model
  is a `Serializer` over an existing dataclass in `mactf/`, and every client
  return value is that dataclass, reconstructed by `from_dict`.
- It does not add a mandatory dependency. `mactf serve` and `mactf.client` use the
  standard library; the only new extra is `asgi`, for an optional adapter.
- It does not ship a WebSocket server. SSE over a WSGI generator response covers
  live progress, and a WebSocket transport would be a separate package.
- It does not pretend the two remaining Gemini threads were read. See
  [00-scope-and-audit.md](00-scope-and-audit.md#2-sources).

## Source material actually used

| Source | Status |
|---|---|
| `docs/framework.md` | Read in full; normative |
| `docs/Designing_Multi_Agent_Incentive_Frameworks.docx` | Extracted and read in full (65,150 chars, 784 lines) |
| `docs/gemini_conversation_28459b25aa1c1b46.docx` | Byte-identical to the above; read in full |
| `https://gemini.google.com/app/28459b25aa1c1b46` | Same content as the docx export |
| `https://gemini.google.com/app/7ac3731679b9b5fd` | Not retrievable anonymously; excluded |
| `https://gemini.google.com/app/8f7dcbb2ed0643e0` | Not retrievable anonymously; excluded |
| `mactf/` source tree | Read in full for the audit in document 00 |
