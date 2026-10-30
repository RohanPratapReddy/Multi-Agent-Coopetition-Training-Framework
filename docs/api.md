# HTTP API

`mactf.contrib.api` serves everything the orchestrator, scorer and registry know over
HTTP. It is an installed app: add `"mactf.contrib.api"` to `INSTALLED_APPS`, leave
`ROOT_URLCONF = None` (or `include("mactf.contrib.api.urls")` in your own conf), and run
`python manage.py serve`. See [apps.md](apps.md). Every path below is relative to
`API.BASE_PATH` (default `api/v1`), and the route **name** is the contract:
`reverse("api:run-detail", run_id="x")` and the OpenAPI `operationId` are the same string.

The route table at the end of this page is generated from
`mactf.http.openapi.schema()`; `GET /api/v1/openapi` returns the same document live.

## Authentication

Requests carry `Authorization: Bearer <token>`. Tokens live in `API.TOKENS`, keyed by
`plain:<secret>` or `sha256:<64 hex>` (use `mactf.contrib.api.auth.hash_token` to
produce the digest, so a settings file never holds a secret), and map to a list of scopes:

| Scope | Allows |
|---|---|
| `read` | every `GET` |
| `run:write` | start, cancel, resume and roll back runs |
| `score:write` | `POST /evaluations`, ledger ingest, adjudications |
| `sandbox:exec` | `POST /sandbox/exec` |
| `admin` | configuration reload and capability listing |
| `worker` | the distributed worker protocol |

With `API.ALLOW_ANONYMOUS = True` (the default) an unauthenticated caller gets `read`
only; set it to `False` to require a token everywhere. A missing or unknown token is
`401`; a valid token without the scope is `403` with `extensions.required_scope`.
`mactf.api.E004` fails `check` on a mistyped scope.

## Errors

Every non-2xx response is `application/problem+json`:

```json
{"type": "urn:mactf:error:not-found", "title": "Not Found", "status": 404,
 "detail": "no run 'x'", "instance": "/api/v1/runs/x", "trace_id": "9f2c...",
 "extensions": {}}
```

`trace_id` is also sent as `X-Trace-Id`. An unexpected exception is a `500` problem
with the trace id and **no** detail or traceback unless `DEBUG` is true. The mapping
the client uses: `400/422` validation, `401` unauthorized, `403` forbidden, `404`
not found, `409` conflict (including a duplicate `run_id`), `413` body too large,
`429` rate limited (`MAX_CONCURRENT_RUNS` reached), `503` capability unavailable.

## Runs and idempotency

`POST /runs` answers `202` with `Location` and a body `{run_id, status, created_at,
links}`. Send `Idempotency-Key`; a repeated key within `IDEMPOTENCY_TTL_SECONDS`
returns the original run instead of starting another, which is what makes the client's
retry of a `POST` safe. `GET`s are retried by the client on `503` and connection
errors, honouring `Retry-After`; a `POST` without a key is never retried.

A run is `queued -> running -> completed | failed | cancelled`, with `paused` and
`resuming` for the watchdog and `POST /runs/{id}/resume`.

## Scorecard versions

`GET .../episodes/{n}/peers/{peer}` (and the episode routes) return the stored v1
scorecard by default. `Accept: application/vnd.mactf.v2+json` (or
`?scorecard_version=2`) returns v2, where `points` is mandatory and `normalization`
and `guards_denied` are added; asking for v2 on a run scored without points is a `409`,
never a silent v1. The header wins over the query string and the response then carries
`Warning: 299`. An unknown version is `406` listing the supported ones.

## Streaming

`GET /runs/{id}/events` is `text/event-stream`. Events have monotonically increasing
integer ids; `epoch` carries `{epoch, score, ...}` and the last frame is `terminal`.
A comment heartbeat is sent every `HTTP.SSE_HEARTBEAT` seconds. Reconnect with
`Last-Event-ID: <n>` and the server replays everything after `n` from a ring buffer of
`API.EVENT_BUFFER` events; if `n` has already been evicted it says so with a `stream_lost`
event naming the oldest retained id, rather than skipping silently. Closing the
connection releases the subscription.

## Routes

| Method | Path | `operationId` | Summary |
|---|---|---|---|
| GET | `/api/v1/admin/capabilities` | `admin-capabilities` | Everything /meta reports plus extras, settings module and writable paths |
| POST | `/api/v1/admin/config/reload` | `admin-config-reload` | Reload settings and rebuild the scorer and run manager |
| POST | `/api/v1/evaluations` | `evaluation-create` | Score a submitted ledger |
| GET | `/api/v1/health` | `health` | Liveness probe |
| GET | `/api/v1/meta` | `meta` | What this server can do |
| GET | `/api/v1/openapi.json` | `openapi` | The OpenAPI 3 document for this server |
| GET | `/api/v1/peers` | `peer-list` | The configured or live peers |
| GET | `/api/v1/peers/{name}` | `peer-detail` | One peer with its contract |
| GET | `/api/v1/peers/{name}/contract` | `peer-contract` | A peer's input/output contract |
| POST | `/api/v1/peers/{path}/infer` | `peer-infer` | Run inference rooted at one node |
| GET | `/api/v1/rules` | `rule-list` | The rule catalog |
| GET | `/api/v1/rules/validation` | `rule-validation` | Self-check of the rule catalog |
| GET | `/api/v1/rules/{rule_id}` | `rule-detail` | One rule |
| GET | `/api/v1/runs` | `run-list` | List runs |
| POST | `/api/v1/runs` | `run-list` | Start a run |
| GET | `/api/v1/runs/{run_id}` | `run-detail` | One run |
| GET | `/api/v1/runs/{run_id}/adjudications` | `run-adjudication-list` | Adjudication history |
| POST | `/api/v1/runs/{run_id}/adjudications` | `run-adjudication-create` | Record an adjudication |
| GET | `/api/v1/runs/{run_id}/aggregations` | `run-aggregation-list` | Aggregation reports of a run |
| POST | `/api/v1/runs/{run_id}/cancel` | `run-cancel` | Request cooperative cancellation |
| GET | `/api/v1/runs/{run_id}/checkpoints` | `run-checkpoint-list` | Checkpoints of a run |
| GET | `/api/v1/runs/{run_id}/checkpoints/{episode}` | `run-checkpoint-detail` | One checkpoint, described and verified |
| POST | `/api/v1/runs/{run_id}/checkpoints/{episode}/rollback` | `run-checkpoint-rollback` | Roll a run back to a checkpoint |
| POST | `/api/v1/runs/{run_id}/checkpoints/{episode}/verify` | `run-checkpoint-verify` | Verify a checkpoint file |
| GET | `/api/v1/runs/{run_id}/episodes` | `run-episode-list` | Episodes of a run |
| GET | `/api/v1/runs/{run_id}/episodes/{episode}` | `run-episode-detail` | One episode |
| POST | `/api/v1/runs/{run_id}/episodes/{episode}/ledger` | `run-episode-ledger` | Append a ledger to a stored episode and rescore that peer |
| GET | `/api/v1/runs/{run_id}/episodes/{episode}/normalization` | `run-episode-normalization` | The cohort normalization report for one episode |
| GET | `/api/v1/runs/{run_id}/episodes/{episode}/peers/{peer}` | `run-episode-peer` | One peer's scorecard for one episode |
| GET | `/api/v1/runs/{run_id}/events` | `run-events` | Live run events (SSE) |
| GET | `/api/v1/runs/{run_id}/flags` | `run-flags` | Flag precision and P19 state |
| POST | `/api/v1/runs/{run_id}/infer` | `run-infer` | Run hierarchical inference over a run's tree |
| GET | `/api/v1/runs/{run_id}/logs/tail` | `run-log-tail` | The last log records of a run |
| GET | `/api/v1/runs/{run_id}/peers/tree` | `run-peer-tree` | The peer tree of a run |
| GET | `/api/v1/runs/{run_id}/peers/{path}` | `run-peer-node` | One node of a run's tree with its latest scorecard |
| GET | `/api/v1/runs/{run_id}/points` | `run-points` | Per-episode, per-peer points for a whole run |
| POST | `/api/v1/runs/{run_id}/resume` | `run-resume` | Continue a paused or interrupted run under the same id |
| POST | `/api/v1/runs/{run_id}/rollback` | `run-rollback` | Roll a run back to the checkpoint of a given episode |
| GET | `/api/v1/runs/{run_id}/rule-counts` | `run-rule-counts` | Rule event counts by rule |
| GET | `/api/v1/runs/{run_id}/rule-events` | `run-rule-event-list` | Individual rule events (the audit surface) |
| GET | `/api/v1/runs/{run_id}/summary` | `run-summary` | A run, its latest episode and its latest aggregation |
| GET | `/api/v1/runs/{run_id}/warnings` | `run-warning-list` | Active warnings per peer and rule |
| GET | `/api/v1/runs/{run_id}/watchdog` | `run-watchdog` | The last health report and recent history |
| POST | `/api/v1/runs/{run_id}/watchdog/snapshot` | `run-watchdog-snapshot` | Take an operator snapshot now |
| POST | `/api/v1/sandbox/exec` | `sandbox-exec` | Run a command inside a peer's sandbox |
| POST | `/api/v1/sandbox/probe` | `sandbox-probe` | Force a fresh probe |
| GET | `/api/v1/sandbox/providers` | `sandbox-provider-list` | Every sandbox provider and what it can do on this host |
| GET | `/api/v1/sandbox/status` | `sandbox-status` | The provider a run would use, and why |
| GET | `/api/v1/scorers` | `scorer-list` | Scorer stage types |
| GET | `/api/v1/scores/scale` | `score-scale` | The 0-100 points scale |
| GET | `/api/v1/severities` | `severity-list` | Severity labels with their configured weights |
| POST | `/api/v1/system-checks` | `system-checks` | Run the system checks |
| GET | `/api/v1/tiers` | `tier-list` | Precedence tiers |
| GET | `/api/v1/trainers/capabilities` | `trainer-capabilities` | Optimizer, loss, scorer and hierarchy capabilities, and which extras are installed |
| GET | `/api/v1/trainers/presets` | `trainer-preset-list` | Named trainer presets from settings |
| POST | `/api/v1/trainers/preview` | `trainer-preview` | Build a trainer's tree and optimizers without training |
| POST | `/api/v1/weights/preview` | `weight-preview` | What would the swarm's weights be for these scores? |
