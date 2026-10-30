# Changelog

All notable changes to `mactf` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[PEP 440](https://peps.python.org/pep-0440/) versions.

## 0.1.0

### Added

- `mactf serve` and the `/api/v1` HTTP API: runs, episodes, scorecards, rule
  events, warnings, aggregations, checkpoints, live SSE streams, served by a
  standard-library WSGI server with no third-party HTTP dependency.
- `mactf.http`: `Request`, `Response`, `JsonResponse`, `StreamingResponse`,
  `ProblemResponse`, `path`/`include`/`reverse`, `View`, middleware,
  `get_wsgi_application()`, declarative `Serializer`/`ModelSerializer`, and an
  OpenAPI generator that reads the URL conf.
- `mactf.contrib.api` as an installed app selected by `INSTALLED_APPS`, exactly as
  `django.contrib.admin` is: `urlpatterns`, views, serializers, bearer-token auth,
  system checks (`mactf.api.E001`-`E004`), and the `serve` command. Removing the
  entry removes the API and nothing else.
- `mactf.client`: `MACTFClient` with `InProcessTransport` (dispatch through
  `ROOT_URLCONF`, no socket) and `HTTPTransport` (stdlib `urllib.request`), both
  returning the framework's own dataclasses and raising the same exception classes.
- `mactf.testing.Client`: the in-process WSGI test client.
- `from_dict` on `PeerScorecard`, `EpisodeScorecard`, `AggregationReport`,
  `CapabilityReport`, `ExecResult`, `ComplianceResult`, and `PointsBreakdown`, so a
  response reconstructs the class that produced it.
- App-contributed management commands, app-scoped migrations, and app-registered
  URL confs (`AppConfig.get_urls()` / `get_commands()` / `get_migrations()`), plus
  the `ROOT_URLCONF`, `MIDDLEWARE`, `APPEND_SLASH`, `HTTP`, and `API` settings.
- `POST /evaluations` and ledger ingest, which make the 14 reward rules and all 23
  penalty rules that had no ingest path reachable from structured evidence.
- Dataset loading from JSONL (`truth`, `ambiguity`, `interaction`), with
  deterministic batching and record-level provenance in task metadata.
- Real bootstrap: behavioural-cloning priming before the RL loop.
- Persistent warnings, adjudications, and aggregation reports (schema v2);
  `--resume` now restores warning state without a checkpoint.
- QEMU guest agent, so `sandbox exec` executes inside the VM.
- Cooperative/competitive score split per framework.md 9.2.
- Distributed runs over HTTP with checksummed parameter exchange
  (`mactf.distributed`).
- `Scorer` is an abstract base class with `RewardScorer`, `PenaltyScorer`, and
  `WarningScorer` children; `ScoringPipeline(Scorer)` is the composite and keeps
  the previous numbers to `1e-12`.
- `mactf.rl.functions`: `ValueFunction`, `AdvantageFunction`, `LossFunction` and the
  supporting wrapper catalogue, each constructible from JSON (`from_spec`).
- `MACTFTrainer` (fan-out over N peers or a `PeerTree`), `MACTFOptimizer` (torch,
  scipy, callable, routed), and `FinetuneConfig` (LoRA/QLoRA/PEFT).
- `PeerStrategy` and `WeightUpdateStrategy` with `KappaSoftmaxAverage` as the
  recorded fallback.
- `MACTFLogger` with typed records and console/file/JSONL/memory sinks.
- `MACTFWatchdog` with sixteen health probes and snapshot/rollback/pause/prune/abort
  actions.
- `MACTFCheckpoint` with a versioned payload, integrity manifest, and
  `resume_from_last_checkpoint()` that restores RNG and data position.
- Arbitrary-depth peer trees (`PeerTree`, `HierarchySpec`), hierarchical scoring
  with non-offsettable `branch_void` propagation, and hierarchical inference.
- A 0-100 rubric score per peer-episode (`final_points`) with one points binding
  per catalog rule, superlinear severity fractions, diminishing returns on repeated
  rewards, 100-point saturation on penalties, and void episodes pinned to `0.0`.
  The docx `final_score` is unchanged and the two are allowed to disagree
  (`framework.md` 9, 9.5.3, 9.5.4 now have code behind them).
- `EpisodeEvidence` and `ClaimAudit` plus a 40-rule guard catalogue, so a reward
  that the framework's eligibility text forbids is denied with a recorded reason and
  the missing evidence fields rather than silently credited.
- Cohort calibration (`CohortStats`, `CohortNormalizer`, `NormalizationReport`) that
  compresses peer spread toward a configurable band while preserving order and
  pinning void peers to `0.0`.
- `COOPETITION.SIGNAL_SOURCE` (`"points"` | `"final_score"`) selecting which economy
  feeds the aggregator, with the kappa rescaling that keeps the entropy target
  meaningful under the bounded signal.
- `GET /scores/scale`, `GET /runs/{id}/points`, and
  `GET /runs/{id}/episodes/{n}/normalization`; scorecard v2 via
  `Accept: application/vnd.mactf.v2+json` with `points` non-optional.
- Client `PointsBreakdown`, `PointsTable`, `NormalizationReport` and the
  `client.scores` namespace.
- `MACTFClient(transport="in_process" | "http", base_url=...)`, `runs.create({...})`
  taking a mapping, and `Event.type` / `Event.epoch`, `Episode.peers`,
  `RunHandle.id` conveniences.
- `PeerScorecard.final_points`, `PeerScorecard.points` readable as attributes, and
  `TrainResult.episodes`, `MACTFTrainer.context`, `scorer.points_table`.
- `MACTFTrainer(model_factory=..., hierarchy=...)`: the factory builds every node
  whose spec has no model of its own, and accepts either `factory(path)` or
  `factory()`.
- New extra `asgi` (optional `uvicorn` for the ASGI adapter), plus `scipy`, `peft`,
  `quant`, `train`, `tensorboard`, `wandb`.

### Fixed

- `SocketChannel.request` now matches response ids to requests, so concurrent or
  out-of-order frames are rejected instead of misread.
- `RunConfig.resume_from` is typed to accept the `bool` the CLI actually passes; the
  annotation previously claimed `Path`.
- `TaskSource.next` derives its seed with `blake2b` instead of builtin `hash()`, so a
  hierarchical run is reproducible across `PYTHONHASHSEED` values.
- `Peer.to` no longer rebuilds the optimizer as Adam after a device move, discarding
  the caller's choice.
- `mactf run` without `--redteam` now defers to `TRAINING.REDTEAM` exactly as an API
  run does, instead of forcing it off; the CLI and the API agree for the same seed.
- A run that has just finished no longer makes the next `POST /runs` answer a
  spurious `429` while the previous run thread is still releasing its slot.
- `ScipyOptimizer` raises `ImproperlyConfigured` naming `pip install "mactf[scipy]"`
  at construction, not at the first step.
- `ResumeReport.bit_identical_expected` depends on the RNG, data, memory and
  optimizer sections only, as documented; sections a run never used (adapters,
  schedulers, grad scaler) are reported but no longer veto it.

### Changed

- `SCHEMA_VERSION` 1 -> 2, then 2 -> 3. `migrate` upgrades in place; no data loss.
  Migration 3 adds `scorecards.points`, `scorecards.points_table_version`,
  `episodes.normalization`, and `aggregations.signal_source`, all nullable or
  defaulted. It does not backfill points; `mactf rescore --points` does that, per
  run, explicitly.
- `Orchestrator.run` accepts optional `should_stop` and `on_epoch`; all existing
  calls are unaffected.
- `PhaseContext` gains `source` and `training_settings` for bootstrap.
- `PeerScorecard` gains `competitive`, `cooperative`, `cooperative_denied`, and
  `branch_void`; existing fields and `final_score` semantics are unchanged.
- `PeerScorecard` gains `points` and `metadata["points"]`; `ScorecardSerializer`
  gains an optional `points` field and `ScorecardV2Serializer` makes it required.
  `AggregationReport` gains `signal_source` and `normalization`.
- `AdaptiveKappa` gains a source-aware initial scale. Under
  `SIGNAL_SOURCE="final_score"` the initial kappa is unchanged; under `"points"` it
  is derived so the initial softmax entropy still lands near the target.
- `MACTFTrainer` now derives points for every episode (never feeding `final_score`),
  and checkpoints periodically through the watchdog's writer when one is supplied.
- `mactf.scoring.pipeline.Scorer` now resolves to `ScoringPipeline` through a
  `DeprecationWarning`-emitting shim; `mactf.scoring.Scorer` is the ABC.
- `local_update` moves behind `TrainerProtocol`; its arithmetic is unchanged and the
  equivalence test proves it.
- `build_peer` gains an optional `optimizer_factory` defaulting to the current Adam,
  so default behaviour is byte-identical.
- New run state `paused`, plus a `resuming` transition, for watchdog and resume
  flows.
- `pyproject.toml` version is `0.1.0`; the release workflow gates tags with
  `packaging.version` and runs seven required claim-specific jobs.

### Deprecated

- `mactf.scoring.pipeline.Scorer` as a constructor. It keeps working through `0.2.x`
  and is removed in `0.3.0`.
- `CheckpointManager` as the recommended writer. It stays as the low-level writer
  and keeps reading format v1.

### Known limits

- A `trainer` spec on `POST /runs` is validated and stored, but orchestrator runs
  started through the API do not execute it; use `MACTFTrainer` directly.
- The aggregator cap makes a single-participant cohort infeasible; a hierarchy needs
  at least two roots.
- `peft` is imported lazily; where it cannot be imported (for instance a broken
  `torchaudio` install) LoRA runs raise `ImproperlyConfigured` naming the extra.
