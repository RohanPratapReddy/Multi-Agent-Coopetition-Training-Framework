---
title: Verification and Release
status: approved-for-implementation
version: 1.0.0
applies_to: mactf >= 0.1.0
---

# 06. Verification and Release

## 1. Test layout

New directories under `tests/`, following the existing flat convention but
grouped by phase so a CI matrix can select them:

```
tests/
  conftest.py                     extended with api_client, client fixtures
  http/
    test_request.py               environ parsing, GET/POST, body, json(), chunked
    test_response.py              JsonResponse, StreamingResponse, ProblemResponse
    test_urls.py                  path, include, converters, resolve, reverse
    test_router.py                APPEND_SLASH, 404 vs 301, namespaces
    test_views.py                 dispatch, method-not-allowed, http_method_names
    test_middleware.py            order in/out, exception translation, request id
    test_handler.py               get_wsgi_application caches per settings object
    test_serializers.py           field family, validation, ModelSerializer, from_dict
    test_exceptions.py            MACTFError -> RFC 9457 problem+json
    test_openapi.py               schema generated from the URL conf
    test_layering.py              criterion 39; http imports no engine module
    test_typing.py                no Any-typed public serializer field
    test_streaming.py             SSE framing, heartbeat, disconnect
  contrib_api/
    conftest.py                   settings fixture, installed/uninstalled apps
    test_meta.py                  health, meta, rules, severities, tiers
    test_runs.py                  create, list, get, summary
    test_run_detail.py            episodes, peers, rule-events, checkpoints
    test_evaluations.py           ad-hoc scoring, ledger ingest, P1 floor
    test_adjudications.py         reversal, W5 expunge, append-only events
    test_flags.py                 precision, P19 activation
    test_weights_preview.py       feasibility, kappa non-advance
    test_sandbox.py               degradation, exec refusal reasons
    test_events.py                SSE ids, Last-Event-ID replay, stream_lost
    test_cancel.py                epoch-boundary stop, 409 on terminal
    test_auth.py                  scopes, 401/403, anonymous limits
    test_errors.py                problem+json for every exception class
    test_pagination.py            limit/offset/has_more
    test_idempotency.py           replay, key scoping
    test_openapi.py               every route named, unique names
    test_contract_completeness.py parses docs/plans/02-api-contract.md
    test_urls.py                  criterion 50; include() and reverse() per name
    test_checks.py                criterion 48; checks only when installed
    test_uninstalled.py           criterion 47; removal breaks nothing else
    test_serve.py                 criterion 49; wsgiref with no third-party HTTP
    test_migrations.py            criterion 51; app-scoped order and idempotence
    test_no_scoring_in_views.py   views contain no scoring arithmetic
    test_resume.py                warnings survive without a checkpoint
    test_trainers.py              /scorers, /trainers/capabilities, /trainers/preview
    test_hierarchy_routes.py      /runs/{id}/peers/tree, /runs/{id}/infer, /peers/{path}/infer
    test_observability.py         /logs/tail, /watchdog, /watchdog/snapshot
    test_resume_route.py          /checkpoints/{episode}/verify, /runs/{id}/resume
    test_points_routes.py         /scores/scale, /runs/{id}/points, normalization
    test_scorecard_v2.py          header and query negotiation, 406, agreement
    test_points_models.py         serializer/policy field-set equality both ways
  client/
    test_in_process.py            criterion 52; full run, no socket, real classes
    test_http.py                  criterion 53; HTTPTransport against mactf serve
    test_parity.py                the parametrized suite over both transports
    test_points_parity.py         document 03.9.1
    test_errors.py                exception mapping on both transports
    test_from_dict.py             round-trip per framework class, missing field raises
    test_no_parallel_models.py    no local dataclass shadows an engine class
    test_layering.py              client imports no engine module at runtime
    test_lazy_import.py           import mactf.client pulls in no torch
  testing/
    test_client.py                criterion 54; no server, no settings leak
  data/
    test_codec.py
    test_jsonl_source.py
    test_synthetic_determinism.py
    test_mixed_source.py
  evaluator/
    test_rule_reachability.py     every R1-R17 and P1-P23
    test_coverage.py              COVERAGE keys == CATALOG keys
    test_no_false_positives.py
    test_warning_eligibility.py
  e2e/
    test_api_cli_parity.py        criterion 1
  distributed/
    test_two_workers.py
    test_scheduling.py
    test_worker_failure.py
    test_transport.py
    test_protocol_roundtrip.py
  test_bootstrap.py
  coopetition/
    test_pool.py
    test_scorecard_split.py
  db/
    test_migrations_v2.py
    test_aggregation_history.py
  sandbox/
    test_guest_agent.py
    test_channel_frames.py
  scoring/
    test_scorer_base.py           ABC, frozen context, describe/state_dict
    test_reward_scorer.py         four buckets, severity scaling, ignored rules
    test_penalty_scorer.py        lambdas, bounded Omega, warning-only exclusion
    test_warning_scorer.py        W1-W9, idempotent prepare, per-node keys
    test_pipeline_equivalence.py  the 1e-12 gate against today's Scorer
    test_pipeline_validation.py   duplicate kind/order, missing kind
    test_shim.py                  pipeline.Scorer warns; scoring.Scorer is the ABC
    test_points_table.py          weights sum to 100, one binding per rule, validation
    test_reward_points.py         diminishing returns, denial recording, determinism
    test_penalty_points.py        superlinear fractions, linear accumulation, saturation
    test_warning_points.py        deduction is always 0.0; notional arithmetic
    test_points_composition.py    gross/clamp/pin, gross survives a void pin
    test_points_monotone.py       criteria 32, 33
    test_guards.py                criterion 34; one row per rule
    test_points_equivalence.py    criteria 35, 38; import independence both ways
    test_points_non_interference.py  points cannot change final_score
    test_normalize.py             criterion 36; the worked example
    test_normalize_invariants.py   N1-N6, permutation invariance
  rl/
    test_value_functions.py
    test_advantage_functions.py
    test_loss_functions.py        CompositeLoss == compute_loss
    test_function_registry.py     from_spec, unknown type/key, capabilities
  training/
    test_optimizer_torch.py
    test_optimizer_scipy.py
    test_optimizer_callable.py
    test_optimizer_router.py      unclaimed parameter raises with its name
    test_finetune.py              LoRA/QLoRA, auto targets, CPU refusal
    test_trainer_build.py         constructor matrix, parameter identity
    test_trainer_fit.py           loop order, hooks, callbacks
    test_trainer_equivalence.py   section 9 gates of document 08
    test_strategies.py
    test_weight_update.py         fallback rule and adjustment text
    test_signal_source.py         entropy convergence, kappa rescaling, resume
    test_import_graph.py          training has no http/contrib_api/client; rl.functions has no training
  logging/
    test_logger_records.py
    test_logger_sinks.py
    test_logger_async.py
    test_logger_redaction.py
    test_logger_signals.py
    test_logger_failure.py        a raising sink cannot fail a step
  watchdog/
    test_policy_validation.py
    test_probes.py
    test_actions.py               snapshot precedes a destructive action
    test_no_mutation.py           no hyperparameter is edited
    test_hierarchy_probes.py
  checkpoints/
    test_payload.py
    test_verify.py
    test_latest_valid.py          corrupt newest is skipped and recorded
    test_resume_ordering.py       optimizer after parameters, RNG after construction
    test_resume_bit_identical.py  criterion 24
    test_resume_warm_restart.py   missing RNG sets bit_identical_expected False
    test_resume_v1.py             CheckpointManager format still readable
    test_prune.py
    test_adapters.py
    test_memory_format.py
  hierarchy/
    test_paths.py
    test_tree_build.py
    test_tree_validate.py
    test_tree_walk.py
    test_flat_equivalence.py      criterion 26
    test_delegation.py
    test_synthesis.py
    test_rollup.py
    test_branch_void.py           criterion 27
    test_warning_scoping.py
    test_train_epoch.py
    test_aggregation_groups.py    criterion 28
    test_inference.py             criteria 29, 30
    test_inference_determinism.py two PYTHONHASHSEED values
    test_inference_scoring.py
    test_failure_isolation.py
    test_prune.py
    test_caps.py
```

The existing 141 tests stay untouched and must continue to pass. Several are
named as regression proofs for the changes in documents 04 and the A0 documents:

| Existing test | Guards |
|---|---|
| `tests/test_e2e.py` | `local_update` relocation in 4.8; flat-tree equivalence in 4.10 |
| `tests/test_rules.py` | catalog immutability (no phase touches rules) |
| `tests/test_continuous.py` | the scorer split must reproduce these numbers to `1e-12` |
| `tests/test_ledger.py` | warning transitions and scorecard void behaviour through the split |
| `tests/test_aggregation.py` | `KappaSoftmaxAverage` must equal `compute_weights` |
| `tests/test_rl.py` | `CompositeLoss` must equal `compute_loss`; advantage wrappers must equal the free functions |
| `tests/test_packaging.py` | new extras do not change metadata invariants |
| `tests/test_sandbox.py` | degradation behaviour preserved by 4.5 |
| `tests/test_peer_adapter.py` | node parameters are never copied by the trainer |
| `tests/test_continuous.py` (points rows) | `final_score` is bit-identical when the points stages are enabled; criterion 35 |
| `tests/test_aggregation.py` (source rows) | `compute_weights` is identical under both `SIGNAL_SOURCE` values when fed the same vector |

## 2. Acceptance criteria

The thirty criteria from document 00 section 7 as originally written, restated as
the test that proves each, followed by the points criteria and the integration
criteria appended in later revisions. A phase is not complete until its rows
pass.

Criteria 31-38 extend the original thirty with the points and calibration
obligations from [11-points-and-normalization.md](11-points-and-normalization.md).
They are appended rather than renumbered because documents 04 and 05 reference
criteria 1-30 by number. Criteria 39-54 extend them again with the transport and
app-integration obligations; they are appended for the same reason.

| # | Criterion | Test | Phase |
|---|---|---|---|
| 1 | API run matches CLI run for the same seed | `tests/e2e/test_api_cli_parity.py` | B |
| 2 | P1 evaluation returns `void` and negative floor | `tests/contrib_api/test_evaluations.py` | B |
| 3 | Every rule ID triggerable | `tests/evaluator/test_rule_reachability.py` | A |
| 4 | Warnings survive resume without a checkpoint | `tests/contrib_api/test_resume.py` | A+B |
| 5 | Both transports return identical framework objects | `tests/client/test_parity.py` | C |
| 6 | Honest sandbox degradation | `tests/contrib_api/test_sandbox.py` | A+B |
| 7 | Guest agent returns real stdout/exit code | `tests/sandbox/test_guest_agent.py` | A |
| 8 | Bootstrap measurably trains | `tests/test_bootstrap.py` | A |
| 9 | SSE replay is gapless | `tests/contrib_api/test_events.py` | B |
| 10 | Cancel stops at an epoch boundary | `tests/contrib_api/test_cancel.py` | B |
| 11 | OpenAPI matches the contract document | `tests/contrib_api/test_contract_completeness.py`, `tests/http/test_openapi.py` | B |
| 12 | Two workers equal single host | `tests/distributed/test_two_workers.py` | D |
| 13 | `ScoringPipeline` equals today's `Scorer` to `1e-12` | `tests/scoring/test_pipeline_equivalence.py` | A0 |
| 14 | Each child scorer equals its `continuous.py` counterpart | `tests/scoring/test_{reward,penalty,warning}_scorer.py` | A0 |
| 15 | A maturing warning is counted once; inverted order raises | `tests/scoring/test_pipeline_validation.py` | A0 |
| 16 | A custom `RewardScorer` cannot bypass the floor | `tests/scoring/test_pipeline_equivalence.py` | A0 |
| 17 | `CompositeLoss` and `GAEAdvantage` equal the free functions | `tests/rl/test_loss_functions.py`, `test_advantage_functions.py` | A0 |
| 18 | `MACTFTrainer()` defaults equal `Orchestrator.run()` | `tests/training/test_trainer_equivalence.py` | A0 |
| 19 | LoRA trains only adapters; unclaimed parameters raise | `tests/training/test_finetune.py`, `test_optimizer_router.py` | A0 |
| 20 | scipy L-BFGS-B converges without gradient accumulation | `tests/training/test_optimizer_scipy.py` | A0 |
| 21 | The weight-update fallback is recorded, never silent | `tests/training/test_weight_update.py` | A0 |
| 22 | JSONL records, ordered tail, sink failure cannot fail a step | `tests/logging/test_logger_async.py`, `test_logger_failure.py` | A0 |
| 23 | A NaN loss snapshots before it aborts | `tests/watchdog/test_actions.py` | A0 |
| 24 | Resume at epoch 3 is bit-identical for epochs 4-6 | `tests/checkpoints/test_resume_bit_identical.py` | A0 |
| 25 | A corrupt newest checkpoint is skipped and recorded | `tests/checkpoints/test_latest_valid.py` | A0 |
| 26 | A depth-1 tree reproduces today's output exactly | `tests/hierarchy/test_flat_equivalence.py` | A0 |
| 27 | A nano-agent P1 voids its root and every ancestor | `tests/hierarchy/test_branch_void.py` | A0 |
| 28 | A three-level tree trains and aggregates per level | `tests/hierarchy/test_train_epoch.py`, `test_aggregation_groups.py` | A0 |
| 29 | Inference is stable across `parallel` and `PYTHONHASHSEED` | `tests/hierarchy/test_inference.py`, `test_inference_determinism.py` | A0 |
| 30 | Budget exhaustion raises with the path, never truncates | `tests/hierarchy/test_inference.py` | A0 |
| 31 | The 17 reward weights sum to exactly 100.0 and every rule has one binding | `tests/scoring/test_points_table.py` | A0 |
| 32 | Points are monotone: a reward never lowers `final_points`, a penalty never raises it | `tests/scoring/test_points_monotone.py` | A0 |
| 33 | A void episode is pinned to `0.0` even when it earned rewards | `tests/scoring/test_points_monotone.py` | A0 |
| 34 | A guard denial is recorded, never silently dropped | `tests/scoring/test_guards.py` | A0 |
| 35 | `final_score` is unchanged by the points path for every fixture ledger | `tests/scoring/test_points_equivalence.py` | A0 |
| 36 | A calibration compresses the spread, preserves order, and pins void peers | `tests/scoring/test_normalize.py`, `test_normalize_invariants.py` | A0 |
| 37 | `GET /scores/scale` and the client agree on all 40 bindings | `tests/contrib_api/test_points_routes.py`, `tests/client/test_points_parity.py` | B+C |
| 38 | `final_score` and `final_points` disagree correctly on a void-with-reward ledger | `tests/scoring/test_points_equivalence.py`, `tests/client/test_points_parity.py` | A0+C |

Criteria 39-54 are the integration obligations. They are stated normatively in
[12-http-core.md](12-http-core.md) section 16 (39-46, the transport) and
[13-app-integration.md](13-app-integration.md) section 10 (47-54, the app, the
client, and the test client); the rows below are the same list, wired to the test
that proves each. Appended rather than renumbered for the same reason as 31-38.

| # | Criterion | Test | Phase |
|---|---|---|---|
| 39 | `import mactf.http` works with no third-party HTTP package installed | `tests/http/test_stdlib_only.py` | B |
| 40 | Every `path(..., name=)` round-trips through `reverse()` and `resolve()` | `tests/http/test_urls.py` | B |
| 41 | `get_wsgi_application()` is callable by `wsgiref` and by `mactf.testing.Client` unmodified | `tests/http/test_handler.py` | B |
| 42 | A view exception yields a `500` problem with a trace id and no traceback unless `DEBUG` | `tests/http/test_middleware.py` | B |
| 43 | Removing the app leaves `mactf/http` importable and `tests/http/` passing | `tests/http/test_layering.py` | B |
| 44 | No module under `mactf/http/` imports an engine module | `tests/http/test_layering.py` | B |
| 45 | The OpenAPI document emits an operation per route with `operationId == name` | `tests/http/test_openapi.py` | B |
| 46 | SSE works through `wsgiref` with heartbeats, and a disconnect releases the subscription | `tests/http/test_streaming.py` | B |
| 47 | Without the app, `run`/`check`/`migrate`/`MACTFClient()` work and `serve` is absent with a helpful error | `tests/contrib_api/test_uninstalled.py` | B |
| 48 | `serve` is discovered from the app, not from a core command package | `tests/contrib_api/test_serve.py` | B |
| 49 | Project overrides of the new settings take effect | `tests/contrib_api/test_settings_overrides.py` | B |
| 50 | `E001`, `W002`, `E003`, and `E004` are registered and reported | `tests/contrib_api/test_checks.py` | B |
| 51 | An app migration applies after core's and records its `(app, version)` row | `tests/contrib_api/test_migrations.py` | B |
| 52 | `from_dict(to_dict(x)) == x` per class, and unknown keys are ignored | `tests/client/test_from_dict.py` | B+C |
| 53 | Both transports return identical objects and identical exception classes | `tests/client/test_parity.py` | C |
| 54 | `mactf.testing.Client` serves without a socket and matches `InProcessTransport` bytes | `tests/testing/test_client.py` | B |

### `tests/e2e/test_api_cli_parity.py` in detail

This is the single most important test in the plan, because it proves the API is
not a parallel implementation. It:

1. Runs `manage.py run --epochs 2 --seed 1337 --device cpu --json` in a
   subprocess with a temp project root, capturing the JSON.
2. Runs the same configuration through `mactf.testing.Client` + `POST /runs` +
   SSE until terminal, capturing episodes. No server is started: the in-process
   path exercises the same URL conf, middleware, views, and serializers.
3. Asserts, per epoch and per peer: `final_score`, `floor`, `void`, `severity`,
   `kappa`, `entropy`, and every weight in the `AggregationReport` are equal to
   `1e-9`.
4. Asserts the two `run_id` values are the only difference.

A third leg repeats step 2 against a real `mactf serve` socket and asserts the
bodies are identical to the in-process ones, which is the only place criterion 1
and criterion 53 meet.

Determinism requires `DETERMINISTIC=True` and `DEVICE=cpu`; the synthetic task
source is already cross-process stable (verified: three subprocesses under
different `PYTHONHASHSEED` values produce identical tensors), so no library fix is
a prerequisite. Criterion 1 can therefore be written without waiting on 4.1, and
4.1 precedes Phase B only because the JSONL sources must be usable by then.

## 3. Fixtures

`tests/contrib_api/conftest.py`:

```python
@pytest.fixture()
def api_settings(tmp_path, monkeypatch) -> Settings:
    """Settings rooted in tmp_path: cpu, 2 peers, no sandbox, sqlite in tmp."""
    # sets MACTF_SETTINGS_MODULE to a generated module, mirrors tests/conftest.py
    # and includes "mactf.contrib.api" in INSTALLED_APPS

@pytest.fixture()
def client(api_settings) -> mactf.testing.Client:
    from mactf.testing import Client
    with Client(settings=api_settings) as c:
        yield c

@pytest.fixture()
def live_server(api_settings):
    """mactf serve on an ephemeral port, for SSE and HTTPTransport tests."""
    # run the management command in a thread or subprocess, poll /api/v1/health,
    # yield base_url, then signal shutdown and join

@pytest.fixture()
def token() -> str: ...
```

`mactf.testing.Client` covers everything except the stream endpoints and
`HTTPTransport`, which need a real socket. `live_server` exists only for those,
which keeps the suite fast. There is no `TestClient` import anywhere in the
suite; `tests/contrib_api/test_uninstalled.py` asserts that by grepping the test
tree for `fastapi` and `httpx`.

## 4. Packaging

### Extras

```toml
[project.optional-dependencies]
gpu = ["numpy>=1.26"]
sandbox = []
shell = ["ipython>=8.18", "bpython>=0.24"]
asgi = ["uvicorn[standard]>=0.30"]
# new: the A0 wrapper layer's optional backends
scipy = ["scipy>=1.11", "numpy>=1.26"]
peft = ["peft>=0.10", "transformers>=4.40"]
quant = ["bitsandbytes>=0.43"]
train = ["scipy>=1.11", "numpy>=1.26", "peft>=0.10", "transformers>=4.40"]
tensorboard = ["tensorboard>=2.16"]
wandb = ["wandb>=0.17"]
docs = ["mkdocs>=1.6", "mkdocs-material>=9"]
dev = [
  "pytest>=8", "pytest-cov>=5", "ruff>=0.6", "mypy>=1.10", "build>=1.2", "twine>=5",
  "tomli>=2.0; python_version < '3.11'",
  "uvicorn[standard]>=0.30",
  "scipy>=1.11", "peft>=0.10", "transformers>=4.40",
]
```

`api`, `sdk`, and `server` are gone. The service and the client are core, so
there is nothing to opt into at the packaging level; opting in is a line in
`INSTALLED_APPS`. `asgi` exists only for the optional adapter in
`mactf/http/asgi.py`, and document 12 requires that adapter to be the only module
in the framework that imports `uvicorn`.

`train` intentionally omits `bitsandbytes`, because it is CUDA-only and would make
a CPU install of `mactf[train]` fail. `quant` is the opt-in for QLoRA. This is the
honest split: a user who wants QLoRA installs `mactf[peft,quant]`.

`dev` gains `uvicorn` and the A0 dependencies so the full suite runs in CI and
for contributors. `tests/test_packaging.py` gains:
- every extra name is in a known set: `{gpu, sandbox, shell, asgi, scipy, peft,
  quant, train, tensorboard, wandb, docs, dev}`
- no extra named `api`, `sdk`, or `server` exists
- `scipy`, `peft`, `quant`, `train`, `tensorboard`, and `wandb` do not appear in
  the base `dependencies`
- `train` contains the `scipy` and `peft` dependencies (subset check)
- `mactf/py.typed` is still packaged
- `mactf/http`, `mactf/contrib/api`, `mactf/client`, `mactf/testing`,
  `mactf/data`, `mactf/distributed`, `mactf/training`, `mactf/rl/functions` all
  appear in the wheel's record (build a wheel in the test the way the existing
  packaging test already does)
- `mactf.training.optimizer` imports with no extras installed, and
  `ScipyOptimizer(...)` raises `ImproperlyConfigured` naming
  `pip install "mactf[scipy]"`

### Wheel content check

The existing wheel-selection rule (`pyproject.toml` `[tool.setuptools.packages.find]`
include `mactf*`) already picks up new subpackages. `tests/test_packaging.py`
must assert the new modules are inside the wheel, because a missing
`mactf/contrib/api/views/__init__.py` would ship a broken service without failing
the build.

## 5. CI

`.github/workflows/deploy.yml` currently has a matrix over Python 3.10-3.13 plus
build, tag gate, TestPyPI/PyPI publish, install smoke test, and GitHub release
(`.github/workflows/deploy.yml`). Changes:

1. Test matrix installs `.[dev]` so all service, client, and A0 tests run.
2. Add a job `api-contract` that runs
   `pytest tests/contrib_api/test_contract_completeness.py tests/http/test_openapi.py tests/contrib_api/test_urls.py`
   and is a required check, because it protects the documents from rot.
3. Add a job `stdlib-only` that installs the wheel with **no** extras and no
   third-party HTTP package (no `fastapi`, no `uvicorn`, no `httpx`, no
   `requests`), then asserts:
   - `python -c "import mactf.http, mactf.client"` succeeds and imports no
     `fastapi`/`uvicorn`/`pydantic`/`httpx`,
   - `python -c "import mactf.client; import sys; assert 'torch' not in sys.modules"`
     succeeds, which is what makes the client lightweight,
   - `mactf serve --help` runs, and `pytest tests/contrib_api/test_serve.py
     tests/http/test_handler.py` passes against a `wsgiref` server.
   This replaces the old `no-heavy-imports` job, which only guarded a package that
   no longer exists.
4. Keep the existing publish jobs unchanged except for the version gate, which
   must accept `0.1.0` as a final release and `rc`/`a`/`b` suffixes as
   pre-releases (the gate already distinguishes by presence of the prerelease
   suffix; verify it against `packaging.version` rather than a string check).
5. Add a job `numeric-equivalence` that runs the section 11 gate of document 07
   and the section 9 gates of documents 08 and 10:
   `pytest tests/scoring/test_pipeline_equivalence.py tests/rl/test_loss_functions.py tests/training/test_trainer_equivalence.py tests/hierarchy/test_flat_equivalence.py`.
   It is a required check because a silent numeric drift is the one failure that
   invalidates every historical scorecard.
6. Add a job `determinism` that runs
   `pytest tests/hierarchy/test_inference_determinism.py tests/checkpoints/test_resume_bit_identical.py`
   twice with `PYTHONHASHSEED=0` and `PYTHONHASHSEED=12345`. It is a required
   check, and it is the only job that would catch a reintroduction of builtin
   `hash()` in seed derivation.
7. Add a job `points-equivalence` that runs
   `pytest tests/scoring/test_points_equivalence.py tests/scoring/test_points_monotone.py tests/scoring/test_normalize_invariants.py tests/scoring/test_points_non_interference.py`.
   It is a required check and is deliberately separate from
   `numeric-equivalence`: the numeric job proves the docx numbers did not move,
   and this job proves the rubric did not reach them. A single job that failed
   would not say which of the two had broken.
8. Add a job `points-parity` that runs
   `pytest tests/contrib_api/test_points_routes.py tests/contrib_api/test_scorecard_v2.py tests/client/test_points_parity.py`.
   It is the only job that exercises a real HTTP server for the points surface, so
   a route that works in-process but not over the wire is caught here rather than
   by a user.
9. Add a job `transport-parity` that runs
   `pytest tests/client/test_parity.py tests/client/test_from_dict.py tests/contrib_api/test_uninstalled.py`.
   It proves the two transports agree and that removing the app removes the API,
   which are the two claims this revision adds; a regression in either is a
   structural failure rather than a wrong number, so it gets its own required
   check.

Windows coverage: the test matrix is currently `ubuntu-latest` only across Python
3.10-3.13 (`.github/workflows/deploy.yml`), while the install smoke test already
covers `ubuntu-latest` and `windows-latest`. Add `windows-latest` for Python 3.12
to the **test** matrix, because the sandbox and path code has real platform
branches (`mactf/sandbox/qemu.py` `WINDOWS`, `mactf/sandbox/local.py` `SYSTEMROOT`
passthrough) that the smoke test does not exercise. Do not add macOS: there is no
`hvf` code path under test and no maintainer on that platform.

## 6. Versioning and release

`mactf/__init__.py:15` is the single source of truth:

```python
__version__ = "0.1.0"
```

`VERSION` parsing is already robust to suffixes
(`re.match(r"\d+(?:\.\d+)*", __version__)`), so `0.1.0` needs no change to that
logic.

Version `0.1.0` gets a new `CHANGELOG.md` at the repository root (not inside
`docs/plans/`, which is a plan archive, not a release record):

```markdown
## 0.1.0

### Added
- `mactf serve` and the `/api/v1` HTTP API: runs, episodes, scorecards, rule
  events, warnings, aggregations, checkpoints, live SSE streams — served by a
  standard-library WSGI server, with no third-party HTTP dependency.
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
  returning the framework's own dataclasses.
- `mactf.testing.Client`: the in-process WSGI test client, replacing
  `fastapi.testclient.TestClient` in the suite.
- `from_dict` on `PeerScorecard`, `EpisodeScorecard`, `AggregationReport`,
  `CapabilityReport`, `ExecResult`, `ComplianceResult`, and `PointsBreakdown`, so
  a response reconstructs the class that produced it.
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
- Distributed runs over HTTP with checksummed parameter exchange.
- `Scorer` is an abstract base class with `RewardScorer`, `PenaltyScorer`, and
  `WarningScorer` children; `ScoringPipeline(Scorer)` is the composite and keeps
  today's numbers to `1e-12`.
- `mactf.rl.functions`: `ValueFunction`, `AdvantageFunction`, `LossFunction` and
  the supporting wrapper catalogue, each constructible from JSON (`from_spec`).
- `MACTFTrainer` (fan-out over N peers or a `PeerTree`), `MACTFOptimizer` (torch,
  scipy, callable, routed), and `FinetuneConfig` (LoRA/QLoRA/PEFT).
- `PeerStrategy` and `WeightUpdateStrategy` with `KappaSoftmaxAverage` as the
  recorded fallback.
- `MACTFLogger` with typed records and console/file/JSONL/memory sinks.
- `MACTFWatchdog` with sixteen health probes and
  snapshot/rollback/pause/prune/abort actions.
- `MACTFCheckpoint` with a versioned payload, integrity manifest, and
  `resume_from_last_checkpoint()` that restores RNG and data position.
- Arbitrary-depth peer trees (`PeerTree`, `HierarchySpec`), hierarchical scoring
  with non-offsettable `branch_void` propagation, and hierarchical inference.
- A 0-100 rubric score per peer-episode (`final_points`) with one points binding
  per catalog rule, superlinear severity fractions, diminishing returns on
  repeated rewards, 100-point saturation on penalties, and void episodes pinned to
  `0.0`. The docx `final_score` is unchanged and the two are allowed to disagree
  (`framework.md` 9, 9.5.3, 9.5.4 now have code behind them).
- `EpisodeEvidence` and `ClaimAudit` plus a 40-rule guard catalogue, so a reward
  that the framework's eligibility text forbids is denied with a recorded reason
  and the missing evidence fields rather than silently credited.
- Cohort calibration (`CohortStats`, `CohortNormalizer`, `NormalizationReport`)
  that compresses peer spread toward a configurable band while preserving order
  and pinning void peers to `0.0`.
- `COOPETITION.SIGNAL_SOURCE` (`"points"` | `"final_score"`) selecting which
  economy feeds the aggregator, with the kappa rescaling that keeps the entropy
  target meaningful under the bounded signal.
- `GET /scores/scale`, `GET /runs/{id}/points`, and
  `GET /runs/{id}/episodes/{n}/normalization`; scorecard v2 via
  `Accept: application/vnd.mactf.v2+json` with `points` non-optional.
- Client `PointsBreakdown`, `RulePoints`, `DenialRecord`, `CohortStats`,
  `NormalizedScore`, `NormalizationReport`, `PointsTable`, and the
  `client.scores` namespace.
- New extra `asgi` (optional `uvicorn` for the ASGI adapter), plus `scipy`,
  `peft`, `quant`, `train`, `tensorboard`, `wandb`.

### Fixed
- `SocketChannel.request` now matches response ids to requests, so concurrent or
  out-of-order frames are rejected instead of misread.
- `RunConfig.resume_from` is typed to accept the `bool` the CLI actually passes;
  the annotation previously claimed `Path`.
- `TaskSource.next` derives its seed with `blake2b` instead of builtin `hash()`,
  so a hierarchical run is reproducible across `PYTHONHASHSEED` values.
- `Peer.to` no longer rebuilds the optimizer as Adam after a device move,
  discarding the caller's choice.

### Changed
- `SCHEMA_VERSION` 1 -> 2, then 2 -> 3. `migrate` upgrades in place; no data loss.
  Migration 3 adds `scorecards.points`, `scorecards.points_table_version`,
  `episodes.normalization`, and `aggregations.signal_source`, all nullable or
  defaulted. It does not backfill points; `mactf rescore --points` does that,
  per run, explicitly.
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
  is derived so the initial softmax entropy still lands near the target. The
  controller's target, bounds, and update rule are untouched.
- `mactf.scoring.pipeline.Scorer` now resolves to `ScoringPipeline` through a
  `DeprecationWarning`-emitting shim; `mactf.scoring.Scorer` is the ABC.
- `local_update` moves behind `TrainerProtocol`; its arithmetic is unchanged and
  the equivalence test proves it.
- `build_peer` gains an optional `optimizer_factory` defaulting to the current
  Adam, so default behaviour is byte-identical.
- New run state `paused`, plus a `resuming` transition, for watchdog and resume
  flows.

### Deprecated
- `mactf.scoring.pipeline.Scorer` as a constructor. It keeps working through
  `0.2.x` and is removed in `0.3.0`.
- `CheckpointManager` as the recommended writer. It stays as the low-level writer
  and keeps reading format v1.
```

`CHANGELOG.md` follows keep-a-changelog headings. `README.md` gains a "Serve and
client" section with a working example (start server, create run, stream events,
read scorecards, and the same calls again through `InProcessTransport` with no
server) and updates "Known limits":

| Existing bullet | Action |
|---|---|
| "No `sandbox exec` without a guest agent" | Remove; the agent ships in provisioned images, and the remaining 127 case is documented under `docs/sandbox.md` |
| "Warnings are per-run, per-peer, per-rule" (persist in checkpoints but not sqlite) | Remove; schema v2 persists them |
| "`JUDGE` is task-verifiable only" | Rewrite: `TaskJudge` is still verifiable-only by design, but `mactf/scoring/evaluator.py` and `POST /evaluations` are now the ingest path for subjective rules |
| "`PRECEDENCE_MODE: strict` is deliberately strict" | Keep unchanged |
| "Trimming drops the dominant peer" | Keep unchanged |

## 7. Rollout

Each phase is independently shippable and independently revertable.

### Phase A - library capability (no new dependencies)

| Step | Deliverable | Exit gate |
|---|---|---|
| A1 | `mactf/data/` + determinism regression test | `tests/data/` green, existing suite green |
| A2 | `mactf/orchestration/bootstrap.py` + settings | `tests/test_bootstrap.py` green |
| A3 | `mactf/scoring/evaluator.py` | `tests/evaluator/` green, criterion 3 |
| A4 | `mactf/db.py` migration 2 + new Registry methods | `tests/db/` green |
| A5 | `mactf/sandbox/guest_agent.py` + provisioning + channel id matching | `tests/sandbox/` green, criterion 7 |
| A6 | `mactf/coopetition/pool.py` + scorecard split | `tests/coopetition/` green |
| A7 | version `0.0.0rc1` -> `0.1.0a1`, build, `twine check`, TestPyPI | publish succeeds |

Phase A is a minor release on its own: six real capabilities with no API
dependency. The alpha tag lets it be published to TestPyPI for user validation
before Phase B lands.

### Phase A0 - extension layer (no new mandatory dependencies)

A0 precedes B because the API must serialize a trainer spec, a peer tree, and a
health report that already exist. It depends on nothing in A.

| Step | Deliverable | Exit gate |
|---|---|---|
| A0.1 | `mactf/scoring/base.py`, `reward_scorer.py`, `penalty.py`, `warning_scorer.py` | `tests/scoring/test_{scorer_base,reward,penalty,warning}.py` green |
| A0.2 | `pipeline.py` `ScoringPipeline` + `Scorer` shim + exports | `test_pipeline_equivalence.py` green at `1e-12`, `test_shim.py` green |
| A0.3 | `mactf/rl/functions/` + registries + `from_spec` | `tests/rl/` green, `CompositeLoss == compute_loss` |
| A0.4 | `mactf/coopetition/tree.py`, `delegation.py`, `synthesis.py` | `tests/hierarchy/test_{paths,tree_build,tree_validate,tree_walk}.py` green |
| A0.5 | `mactf/scoring/hierarchy.py` + `branch_void` | `test_flat_equivalence.py` (criterion 26) and `test_branch_void.py` (criterion 27) green |
| A0.6 | `mactf/training/{optimizer,protocol,finetune}.py` | `tests/training/test_optimizer_*.py`, `test_finetune.py` green |
| A0.7 | `mactf/training/trainer.py` + `mactf/coopetition/{strategy,weights}.py` | `test_trainer_equivalence.py` (criterion 18), `test_weight_update.py` (criterion 21) green |
| A0.8 | `mactf/training/logging.py` | `tests/logging/` green, criterion 22 |
| A0.9 | `mactf/training/watchdog.py` + `orchestration/checkpoint_v2.py` | `tests/watchdog/` (criterion 23), `tests/checkpoints/` (criteria 24-25) green |
| A0.10 | inference (`run_inference`, `MACTFTrainer.infer`) + caps | `test_inference*.py` (criteria 29-30), `test_caps.py` green |
| A0.11 | version `0.1.0a2`, build, `twine check`, TestPyPI | publish succeeds, `stdlib-only` still green |

The order inside A0.1-A0.2 is not negotiable: the equivalence gate must be green
before `hierarchy.py`, and both gates before the trainer. A0.3 can run in parallel
with A0.4, since the function wrappers and the tree do not import each other.

A0.12-A0.15 are the points layer, and they land after A0.10 rather than beside
A0.1 because 4.11 extends the pipeline A0.2 built and reads the `branch_void`
argument A0.5 added.

| Step | Deliverable | Exit gate |
|---|---|---|
| A0.12 | `mactf/scoring/points.py`, `evidence.py` | `test_points_table.py`, `test_evidence.py` green; weights sum to `100.0` |
| A0.13 | `mactf/scoring/guards.py` + `points()` on the three scorers | `test_guards.py` (criterion 34) and the three `*_points.py` files green |
| A0.14 | pipeline composition, void pin, `score_episode_normalized` shape | criteria 32, 33, 35, 38 green; `test_points_non_interference.py` green |
| A0.15 | `mactf/scoring/normalize.py`; `SIGNAL_SOURCE`; kappa rescaling | criterion 36 green; `test_signal_source.py` entropy convergence green |
| A0.16 | version `0.1.0a3`, build, `twine check`, TestPyPI | publish succeeds; the `points-equivalence` CI job green |

A0.14's gate is the one that must not be skipped: it proves `final_score` did not
move before A0.15 starts compressing `final_points`. A calibration layer on top of
an unverified rubric is two unknowns multiplied.

### Phase B - HTTP layer and service app

| Step | Deliverable | Exit gate |
|---|---|---|
| B1 | `run.py` `should_stop`/`on_epoch`; `orchestration/live.py` `RunManager`, `EventBus` | `tests/contrib_api/test_cancel.py`, `test_events.py` green |
| B2 | `mactf/apps.py` extensions, `command_packages()` from apps, app-scoped migrations, new `DEFAULTS` keys | criteria 47-51 prerequisites green |
| B3 | `mactf/http/`: request, response, urls, router, views, middleware, handler | criteria 39-43, 45 green |
| B4 | `mactf/http/`: serializers, exceptions, openapi | criteria 44, 46 green |
| B5 | `mactf/contrib/api/`: `apps.py`, `urls.py`, `views/`, `serializers.py`, `auth.py`, `checks.py` | `test_meta.py`, `test_auth.py`, `test_errors.py`, `test_urls.py`, `test_checks.py` green |
| B6 | run and detail routes | `test_runs.py`, `test_run_detail.py` green |
| B7 | evaluations, adjudications, flags | `test_evaluations.py`, `test_adjudications.py`, `test_flags.py` green |
| B8 | SSE streams | `test_events.py` green |
| B9 | `mactf serve` on `wsgiref`, `mactf.testing.Client` | criteria 49, 54; `test_contract_completeness.py` green |
| B10 | `from_dict` on the nine framework classes of doc 13 section 5 | `tests/client/test_from_dict.py` green |
| B11 | wrapper-layer routes: `/scorers`, `/trainers/capabilities`, `/trainers/preview`, `/runs/{id}/peers/tree`, `/runs/{id}/infer`, `/peers/{path}/infer`, `/runs/{id}/logs/tail`, `/runs/{id}/watchdog`, `/runs/{id}/resume`, `/scores/scale`, `/runs/{id}/points`, `/runs/{id}/episodes/{n}/normalization` | `test_trainers.py`, `test_hierarchy_routes.py`, `test_observability.py`, `test_resume_route.py`, `test_points_routes.py`, `test_scorecard_v2.py` green |
| B12 | parity test | criterion 1 |
| B13 | version `0.1.0b1`, TestPyPI | publish succeeds, `stdlib-only` job green |

B2 precedes B3 deliberately. If `mactf/http` landed first it would be importable
but unmountable, and the first app that used it would discover the missing
`ROOT_URLCONF` key by failing at request time rather than at `check` time.

### Phase C - client

| Step | Deliverable | Exit gate |
|---|---|---|
| C1 | `base.py` (`Transport` protocol and the `RunHandle`/`Event`/`Page`/`Run`/`Episode`/`RuleEvent` envelopes), `errors.py`, `response` parsing | `tests/client/test_errors.py` green |
| C2 | `InProcessTransport` | `test_in_process.py` green, criterion 52 |
| C3 | `HTTPTransport` + SSE parser + retries | `test_http.py` green, criterion 53 |
| C4 | `MACTFClient` namespaces incl. `runs`, `scores`, `peers`, `sandbox` | `test_parity.py` green, criterion 5; `test_points_parity.py` green, criteria 37-38 |
| C5 | version `0.1.0rc1`, TestPyPI | publish succeeds |

### Phase D - distributed

| Step | Deliverable | Exit gate |
|---|---|---|
| D1 | `protocol.py`, `transport.py` | `test_transport.py`, `test_protocol_roundtrip.py` green |
| D2 | coordinator + `urls.py`/`apps.py` included by the api app | `test_scheduling.py` green |
| D3 | worker + `local_step` extraction | `test_two_workers.py` green, criterion 12 |
| D4 | failure handling | `test_worker_failure.py` green |
| D5 | version `0.1.0`, PyPI | full publish pipeline green |

### Release gates for `0.1.0` final

All fifty-four acceptance criteria pass, `pytest tests -q` is green on the full
matrix, `twine check dist/*` reports PASSED for both artifacts, the `stdlib-only`
job is green, the `numeric-equivalence`, `points-equivalence`, `determinism`,
`points-parity`, and `transport-parity` jobs are green, and the install smoke test
succeeds from PyPI.

## 8. Documentation obligations

The plans in this directory describe what to build. Shipping requires user-facing
docs that do not assume the reader has read them:

| Document | Content |
|---|---|
| `README.md` | install, `mactf serve`, client quickstart with both transports, link to the API docs page |
| `docs/api.md` | generated from `mactf.http.openapi` plus prose for auth, errors, streaming |
| `docs/client.md` | the four recipes: run a swarm, stream epochs, score a ledger, preview weights, each shown over both transports |
| `docs/apps.md` | how `mactf.contrib.api` is installed and removed, what `AppConfig` may declare, how a project's `urls.py` mounts the app |
| `docs/sandbox.md` | guest agent install, image requirements, capability matrix |
| `docs/distributed.md` | coordinator/worker setup, network requirements, failure behaviour |
| `docs/training.md` | `MACTFTrainer`, `MACTFOptimizer`, `FinetuneConfig`, and the four recipes: custom reward scorer, custom loss, LoRA fine-tune, scipy optimizer |
| `docs/hierarchy.md` | `PeerTree`, `HierarchySpec`, delegation and synthesis, hierarchical inference, non-offsettable propagation |
| `docs/scoring.md` | the 0-100 rubric: the points table, why rewards diminish and penalties do not, what a denial means, what void pins to, and the explicit statement that `final_score` and `final_points` answer different questions |
| `docs/calibration.md` | `TARGET_MEAN` and `SHRINK` in plain language, the worked `20/40/60/80 -> 51.5/58.5/65.5/72.5` example, why void peers are excluded from the cohort mean, and how to choose `SIGNAL_SOURCE` |
| `docs/recovery.md` | checkpoints, the payload sections, `resume_from_last_checkpoint`, what "bit identical" does and does not promise |
| `docs/observability.md` | `MACTFLogger` sinks and records, `MACTFWatchdog` probes and actions, API routes for both |
| `CHANGELOG.md` | as in section 6 |

`docs/framework.md` is not edited. If a reader needs to know how scoring works,
they are sent there, because there must be exactly one normative scoring
document.

## 9. Definition of done

The work in this plan is complete when, on a clean checkout:

```powershell
pip install -e ".[dev]"
python -m pytest tests -q
python -m build
python -m twine check dist/*
python training\manage.py check
python training\manage.py run --epochs 1 --device cpu
mactf serve --port 8010        # in one shell
```

and a script running against that server can, in order: read `/meta`, load
`/rules`, `POST /runs`, `GET` the SSE stream to terminal, read every episode's
scorecard, `POST /evaluations` with a P1 event and observe `void: true`, call
`POST /weights/preview`, and read `/sandbox/status` - with every response
matching the serializers in document 02 and every error case returning
`application/problem+json`.

The same sequence must then work with no server at all, in one process:

```python
from mactf.client import MACTFClient
from mactf.scoring import PeerScorecard

client = MACTFClient(transport="in_process")     # dispatches through ROOT_URLCONF
run = client.runs.create({"epochs": 1, "seed": 1337})
for event in client.runs.stream(run.id):
    print(event.type, event.epoch)
card = client.runs.episode(run.id, 1).peers["peer-a"]
assert isinstance(card, PeerScorecard)           # the framework's own class
```

and again against `mactf serve --port 8010` by changing one argument to
`transport="http", base_url="http://127.0.0.1:8010"`, with the same objects
returned. That equality is the definition of done for the integration: the
package a user installs serves itself, and the client is the same program with
and without a socket.

Additionally, on the same checkout, this script must succeed end to end:

```python
from mactf.training import MACTFTrainer
from mactf.coopetition.tree import HierarchySpec

trainer = MACTFTrainer(
    model_factory=build_model,
    hierarchy=HierarchySpec.from_spec({
        "roots": 2,
        "levels": [{"label": "subagent", "count": 2},
                   {"label": "miniagent", "count": 2}],
    }),
    peer_strategy="hierarchical",
    weight_update="kappa_softmax",
)
trainer.build()
result = trainer.fit(epochs=2)
report = trainer.checkpoint.resume_from_last_checkpoint(trainer=trainer)
assert report.bit_identical_expected
out = trainer.infer({"input_ids": sample}, node_path=None)
health = trainer.watchdog.tick(trainer.context)
assert health.status in {"healthy", "degraded"}
```

That script is the definition of done for the wrapper layer: a hierarchy of eight
nodes trains, aggregates level by level, checkpoints with a verifiable manifest,
resumes with RNG and data position restored, serves an inference through the same
tree, and reports health. If any of those steps needs a private helper or a manual
call, the layer is not done.

The same script must then be able to read the rubric without importing anything
private:

```python
scale = trainer.scorer.points_table            # the PointsTable in force
card = result.episodes[-1].peers["peer-0"]
assert card.metadata["points"]["final_points"] == pytest.approx(card.final_points)
assert 0.0 <= card.points.final_points <= 100.0
normalized = trainer.normalize_episode(result.episodes[-1])
assert normalized.preserves_order
assert all(0.0 <= s.calibrated <= 100.0 for s in normalized.scores)
```

`trainer.normalize_episode` is a one-line delegation to
`CohortNormalizer.normalize` over that episode's scorecards, present so a user can
ask the calibration question without constructing a normalizer and a policy. It
adds no arithmetic of its own, which is the same rule every wrapper in this plan
follows.
