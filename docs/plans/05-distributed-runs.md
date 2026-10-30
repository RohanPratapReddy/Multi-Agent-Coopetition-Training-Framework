---
title: Distributed Runs
status: approved-for-implementation
version: 1.0.0
applies_to: mactf >= 0.1.0
---

# 05. Distributed Runs

## 1. Scope and default

Distributed mode lets peers train on more than one host. The coordinator owns
scoring and aggregation; workers own forward/backward passes. This mirrors
docx Turn 8's "distributed training repository" intent without adopting its
Docker-and-YAML execution model, which the framework replaced with QEMU
isolation in Turn 9.

**Single-host is the default and must remain fully functional.** Distributed mode
activates only when at least one worker registers. `POST /runs` does not grow a
required `distributed` field; the coordinator decides based on live membership.

## 2. Why HTTP and not gRPC

1. The framework already ships an HTTP transport with no third-party dependency
   (document [12](12-http-core.md)). Adding `grpcio` plus a `.proto` toolchain
   to a training framework is a large dependency for one internal channel.
2. Workers are few (one per host) and messages are large but infrequent (one
   parameter exchange per epoch).
3. The parameter vector is already a flat `torch.float32` tensor from
   `PeerAdapter.export_parameters()` (`mactf/coopetition/adapter.py:274`). It
   serializes with `torch.save` to bytes and travels in an HTTP body without
   conversion, which gRPC would also require.

The cost is that HTTP/1.1 has no multiplexing, so each worker's request is
sequential. With one exchange per epoch and epochs measured in seconds, that is
not a bottleneck worth a new dependency.

## 3. Modules

```
mactf/distributed/
  __init__.py       Coordinator, Worker, protocol dataclasses
  apps.py           DistributedConfig(name="mactf.distributed", label="distributed")
  protocol.py       WorkerRegistration, WorkerHeartbeat, TaskAssignment,
                    TaskResult, ParameterExchange
  transport.py      encode/decode parameter vectors, checksum, compression
  coordinator.py    Coordinator: membership, sharding, barrier, aggregation
  worker.py         Worker (loop), WorkerClient (client side)
  urls.py           urlpatterns under worker/v1, included by the api app
  views.py          the six worker routes as mactf.http View classes
```

`mactf/distributed/` is importable without the api app; only `views.py` and
`urls.py` import `mactf.http`. A worker process that only runs `Worker` therefore
never constructs a WSGI application.

### `protocol.py`

```python
@dataclass(frozen=True)
class WorkerRegistration:
    worker_id: str
    hostname: str
    capabilities: Mapping[str, Any]       # device, dtype, torch version
    max_peers: int = 1

@dataclass(frozen=True)
class WorkerHeartbeat:
    worker_id: str
    epoch: int
    peers: tuple[str, ...]
    load: float
    free_memory_bytes: int

@dataclass(frozen=True)
class TaskAssignment:
    run_id: str
    epoch: int
    task_id: str
    states_shape: tuple[int, ...]
    targets_shape: tuple[int, ...]
    peer_names: tuple[str, ...]
    payload_b64: str                  # torch.save of {"states", "targets"}
    timeout: float

@dataclass(frozen=True)
class TaskResult:
    task_id: str
    worker_id: str
    peer_names: tuple[str, ...]
    parameters_b64: str
    checksum: str
    metrics: Mapping[str, float]      # local loss, grad norm, duration
    ok: bool
    error: str | None = None

@dataclass(frozen=True)
class ParameterExchange:
    run_id: str
    epoch: int
    parameters_b64: str
    checksum: str
    reference_b64: str | None
    kappa: float
```

All dataclasses have `to_dict`/`from_dict` using plain JSON. `payload_b64` and
`parameters_b64` are base64 of `torch.save(..., file=BytesIO)`.

### `transport.py`

```python
def encode_tensors(payload: Mapping[str, torch.Tensor], *, compress: bool = True) -> tuple[str, str]:
    """Return (base64, blake2b hex checksum) of a torch.save blob."""

def decode_tensors(b64: str, checksum: str, *, expected: Sequence[tuple[str, tuple[int, ...]]] | None = None) -> dict[str, torch.Tensor]:
    """Verify checksum and, when given, every tensor's shape."""

def vector_checksum(vector: torch.Tensor) -> str: ...
```

Checksums are mandatory rather than optional because a truncated response body
over a flaky link must fail loudly. `decode_tensors` raises
`ParameterIntegrityError(MACTFError)` naming the mismatched tensor and both
shapes, so a worker running a different peer layout is diagnosed immediately
rather than producing a subtly wrong average.

Compression: `zlib.compress` at level 6 on the serialized bytes before base64.
Typical float32 parameter vectors compress poorly (20-30%), but the CPU cost is
negligible next to the network transfer, and a no-compression path exists via
`compress=False`.

## 4. Coordinator

```python
class Coordinator:
    def __init__(
        self,
        *,
        settings: Settings,
        registry: Registry,
        heartbeat_timeout: float = 30.0,
        max_workers: int = 16,
    ) -> None: ...

    # membership
    def register(self, registration: WorkerRegistration) -> Mapping[str, Any]: ...
    def heartbeat(self, heartbeat: WorkerHeartbeat) -> Mapping[str, Any]: ...
    def deregister(self, worker_id: str) -> None: ...
    def members(self) -> list[Mapping[str, Any]]: ...
    def reap_stale(self) -> list[str]: ...

    # epoch orchestration
    def plan(self, epoch: int, peers: Sequence[str]) -> dict[str, list[str]]: ...
    def wait_for_results(self, epoch: int, expected: int, *, timeout: float) -> list[TaskResult]: ...
    def submit(self, worker_id: str, assignment: TaskAssignment) -> None: ...
    def aggregate(self, results: Sequence[TaskResult], report: AggregationReport) -> AggregateResult: ...
```

### Sharding

`plan` assigns peers to workers by round-robin over live members sorted by
`worker_id`, respecting each worker's `max_peers`. A swarm of 3 peers and 2
workers yields `{w1: [p1, p2], w2: [p3]}`. Assignment is deterministic given the
same membership, so two coordinators with the same view produce the same plan.

A peer assigned to no worker is not silently dropped: `plan` raises
`DistributedError` naming the unassigned peers and the total capacity. This is the
distributed analogue of `require_same_layout` failing loudly.

### Barrier

`wait_for_results(epoch, expected, timeout)` blocks until `expected` results for
that epoch arrive or the timeout expires. It is a `threading.Condition` over a
dict of `(epoch, task_id) -> TaskResult`.

On timeout:

1. Every missing worker is marked stale and reaped.
2. The epoch proceeds with the results it has.
3. Each missing peer is treated as **not participating** for that epoch:
   `eligible=False` for weight construction, so it contributes zero weight rather
   than a default. This is framework.md 10.1.4 applied to hosts instead of
   episodes, and it is the rule that prevents a dead host from silently holding
   its share.
4. An `integrity_incident` signal fires with the worker ids and the missing peers.

### Aggregation

The coordinator, not the worker, calls `Aggregator.compute_weights` and
`Aggregator.aggregate`. Reasons:

- Weights depend on scores, and scores come from the coordinator's `Scorer`.
  Moving weight computation to workers would require shipping every scorecard to
  every worker and would create `N` opportunities for divergent kappa.
- `AdaptiveKappa` is stateful and advances once per epoch
  (`mactf/coopetition/temperature.py:112`). One owner, one advance.

Workers receive the aggregated vector and load it. They never compute a weight.

```python
def aggregate(self, results, report) -> AggregateResult:
    adapters = {peer: RemoteAdapter(vector=result.parameters) for ...}
    return self.aggregator.aggregate(adapters, report, reference=self.reference_adapter)
```

`RemoteAdapter` is a small shim exposing `export_parameters()` and
`parameter_shapes()` so `Aggregator.aggregate` accepts worker-provided vectors
without a real `PeerAdapter`. It lives in `coordinator.py`, not in
`mactf/coopetition/`, because it exists only for distribution.

## 5. Worker

```python
class Worker:
    def __init__(
        self,
        *,
        coordinator_url: str,
        token: str | None = None,
        device: str = "auto",
        max_peers: int = 1,
        heartbeat_interval: float = 5.0,
    ) -> None: ...

    def run_forever(self) -> None: ...
    def run_epoch(self, assignment: TaskAssignment, peers: dict[str, Peer]) -> TaskResult: ...
    def shutdown(self) -> None: ...
```

Loop:

1. `POST /worker/v1/register` with capabilities from `Runtime.describe()`.
2. Background thread sends `POST /worker/v1/heartbeat` every
   `heartbeat_interval`.
3. `GET /worker/v1/tasks?worker_id=...` long-polls for an assignment
   (server holds the request up to 20s).
4. On assignment: decode tensors, build or reuse the peers listed in
   `peer_names`, run the local update for the assigned peers (`local_update`
   logic from `mactf/orchestration/phases.py:142`, extracted into a reusable
   function `run_local_step(peer, states, targets, signal, loss_config)` so both
   paths share it), return `parameters_b64`.
5. On `ParameterExchange`, load the aggregated vector into every local peer and
   the reference.
6. `POST /worker/v1/deregister` on shutdown.

Extraction note: `local_update` currently contains the whole per-peer loop.
The refactor pulls the body into a module-level function that takes one peer and
returns the loss breakdown; `local_update` becomes a loop over that. Behaviour is
identical and the existing `test_e2e.py` proves it.

Worker peer construction uses the same `mactf.apps._default_modules` /
`build_peer` path as the coordinator, so a peer class must be importable on every
host. That is documented as a deployment requirement: distribute the project
package, not just its configuration.

## 6. Routes

The distributed routes live in `mactf/distributed/urls.py` and are mounted by the
**project's** `ROOT_URLCONF` at a prefix the operator chooses (conventionally
`/worker/v1`). The api app does not include them, because an app that hard-codes
another app's prefix makes the second app unremovable and unrelocatable:

```python
# mactf/distributed/urls.py
from mactf.http.urls import path
from mactf.distributed import views

app_name = "worker"
urlpatterns = [
    path("register", views.register, name="register"),
    path("heartbeat", views.heartbeat, name="heartbeat"),
    path("tasks", views.tasks, name="tasks"),
    path("results", views.results, name="results"),
    path("exchange", views.exchange, name="exchange"),
    path("deregister", views.deregister, name="deregister"),
]
```

```python
# project/urls.py
from mactf.http.urls import include, path

urlpatterns = [
    path("api/v1/", include("mactf.contrib.api.urls")),
    path("worker/v1/", include("mactf.distributed.urls")),   # only if installed
]
```

**Why `include()` and not a conditional in a factory.** `mactf.distributed` is an
app with its own `AppConfig` and `urls_module`. If it is not in
`INSTALLED_APPS`, the project simply does not write the second `include()`, so
`reverse()` raises for its route names and `resolve()` returns 404, with no `if`
in the api app and no `DISTRIBUTED` flag threaded through the transport. That is
the same mechanism the api app itself uses, applied one level up.

The scope is `worker`, not `read` or `run:write`, because a worker credential
should not be able to read scorecards or start runs.

| Method | Name | Path | Scope | Purpose |
|---|---|---|---|---|
| POST | `worker:register` | `/worker/v1/register` | `worker` | register, returns `worker_id` and the run's peer layout |
| POST | `worker:heartbeat` | `/worker/v1/heartbeat` | `worker` | liveness plus load |
| GET | `worker:tasks` | `/worker/v1/tasks` | `worker` | long-poll for an assignment |
| POST | `worker:results` | `/worker/v1/results` | `worker` | submit a `TaskResult` |
| POST | `worker:exchange` | `/worker/v1/exchange` | `worker` | fetch the aggregated vector for an epoch |
| POST | `worker:deregister` | `/worker/v1/deregister` | `worker` | leave cleanly |

`worker` is a new scope, distinct from `read`, `run:write`, `score:write`,
`sandbox:exec`, and `admin`, because a worker credential should not be able to
read scorecards or start runs. Its validity is enforced by `mactf.api.E004`, the
same check every other scope goes through (document 13 section 2.3).

`RegistrationRequest` validation: `worker_id` must be
unique; duplicate registration from the same hostname with a different id is
allowed (two processes per host), but the same id with a different hostname is
rejected with 409 because it indicates a spoofed or recycled identity.

When the distributed app is not installed the routes are absent, and
`tests/contrib_api/test_contract_completeness.py` handles that by testing the
default `INSTALLED_APPS` and a distributed-configured set separately: the first
must not contain `worker-*` names, the second must.

## 7. Failure semantics

| Failure | Detection | Behaviour |
|---|---|---|
| Worker misses heartbeats | `reap_stale()` after `heartbeat_timeout` | Removed from membership; its in-flight task is discarded; its peers get `eligible=False` for the epoch |
| Worker dies mid-task | Barrier timeout | Same as above; the epoch completes with partial peers |
| Checksum mismatch | `decode_tensors` | `ParameterIntegrityError`; that worker's result is rejected; epoch proceeds without it |
| Worker returns wrong peer set | `TaskResult.peer_names` compared to the assignment | Result rejected, worker marked `suspect`, an `integrity_incident` fires |
| Coordinator restarts | Workers see 404 on heartbeat | Worker re-registers with the same `worker_id`; the coordinator treats it as new; the current run is `interrupted` and unrecoverable for that epoch |
| Split brain (two coordinators) | Not detected | Out of scope, documented. Requires an external lock; sqlite is single-writer so a second coordinator cannot share the registry file safely, which is the practical protection |

Rule: **a dropped worker contributes zero weight, never a default.** This is
framework.md 10.1.4 ("A peer that did not complete an episode ... contributes
zero weight for that episode rather than a default weight") applied to the
transport. It is asserted in `tests/distributed/test_worker_failure.py`.

## 8. Settings

Distribution is enabled by installing `mactf.distributed` in `INSTALLED_APPS`,
not by a flag. The settings below are read by its `AppConfig` and by the
coordinator; absent app, absent routes, absent settings validation.

```python
"DISTRIBUTED": {
    "COORDINATOR_URL": "http://127.0.0.1:8000",
    "WORKER_ID": None,               # generated from hostname+pid when null
    "MAX_PEERS": 1,
    "COMPRESS": True,
    "WORKER_HEARTBEAT_TIMEOUT": 30.0,
    "WORKER_TASK_TIMEOUT": 600.0,
    "MAX_WORKERS": 16,
},
```

A single `DISTRIBUTED` block rather than the old split between `API` and
`DISTRIBUTED`: the heartbeat and worker-limit values were never HTTP concerns,
and keeping them under `API` made the api app appear to own distributed state it
does not read.

## 9. Explicit non-goals

- **No parameter sharding.** Each worker holds complete peer models. Sending
  partial parameter vectors would make the aggregation average a mix of stale and
  fresh parameters, which framework.md 10.1 does not describe and which cannot be
  audited.
- **No gradient averaging.** MACTF aggregates parameters after local updates by
  design (docx Turn 4/7: "weighted sum of all parameters"). Distributed gradient
  averaging would be a different algorithm.
- **No fault-tolerant replay of a lost epoch.** A missed epoch is reported and the
  run continues. Replaying would require storing every worker's inputs.
- **No cross-coordinator peer discovery.** Static `COORDINATOR_URL`.
- **No TLS termination.** Deploy behind a reverse proxy. Documenting this is
  better than a half-configured TLS flag.

## 10. Tests

`tests/distributed/test_two_workers.py`: start the app on an ephemeral port, run
two `Worker` instances in threads against a 2-peer swarm for one epoch, and assert
the aggregated parameter vector equals the single-host vector for the same inputs
within `1e-9`. This is acceptance criterion 12.

`tests/distributed/test_scheduling.py`: `plan` is deterministic, respects
`max_peers`, and raises on unassigned peers.

`tests/distributed/test_worker_failure.py`: kill one worker mid-epoch; the run
completes, the dead worker's peer has weight `0.0` in the persisted
`AggregationReport`, and an `integrity_incident` event appears in the SSE stream.

`tests/distributed/test_transport.py`: checksum tampering raises
`ParameterIntegrityError`; shape mismatch names both shapes; compression is
lossless for float32 including `-0.0` and `inf`.

`tests/distributed/test_protocol_roundtrip.py`: every protocol dataclass survives
`to_dict`/`from_dict` unchanged.
