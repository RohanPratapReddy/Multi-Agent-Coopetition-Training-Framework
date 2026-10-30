# Distributed runs

A coordinator owns scoring and aggregation; workers own the forward and backward pass. With no
live worker the single-host path runs untouched, so this is purely additive.

## Pieces

- `mactf.distributed.Coordinator`: membership, sharding peers across workers, the epoch barrier,
  aggregation. Installed on a run as its `distributor`; it is active only while at least one worker
  is live.
- `mactf.distributed.Worker` and `WorkerClient`: register, heartbeat, poll for a `TaskAssignment`,
  run the epoch, post a `TaskResult`, then fetch and apply a `ParameterExchange`.
- Wire dataclasses (`protocol.py`) are frozen, JSON-only and torch-free; tensors travel as base64
  strings from `encode_tensors` with a checksum, optionally compressed. A checksum mismatch raises
  `ParameterIntegrityError` rather than applying corrupt weights.

## Setup

Install the API app and the distributed app on the coordinator host (the routes need the API):

```python
INSTALLED_APPS = [..., "mactf.contrib.api", "mactf.distributed"]
API = {"ALLOW_ANONYMOUS": False, "TOKENS": {"plain:w0rker": ["worker"]}}
DISTRIBUTED = {"MAX_WORKERS": 8, "WORKER_HEARTBEAT_TIMEOUT": 30.0, "WORKER_TASK_TIMEOUT": 600.0}
```

```powershell
python manage.py serve --port 8000
```

On each worker host:

```python
from mactf.distributed import Worker

worker = Worker(coordinator_url="http://coordinator:8000", token="w0rker", device="cuda", max_peers=2)
worker.run_forever()
```

Routes live under `DISTRIBUTED.BASE_PATH` (default `worker/v1`) and need the `worker` scope.

## Settings

`COORDINATOR_URL`, `BASE_PATH`, `WORKER_ID`, `MAX_PEERS`, `COMPRESS` (default true),
`WORKER_HEARTBEAT_TIMEOUT`, `WORKER_TASK_TIMEOUT`, `MAX_WORKERS` (default 16). `check` reports
mistyped values as `mactf.distributed.E001`.

## Failure behaviour

A worker that misses heartbeats for `WORKER_HEARTBEAT_TIMEOUT` is reaped. If tasks are still
outstanding after `WORKER_TASK_TIMEOUT`, every worker holding one is reaped and its peers are
reported as missing; the epoch proceeds with the results that arrived, and the missing peers keep
their coordinator-side parameters for that epoch (the report names them). Tasks are not
re-sent to another worker within the epoch. While no worker is live the run uses the single-host
path. The coordinator is a single process (as is the API), so it is a single point of failure;
checkpoints ([recovery.md](recovery.md)) are how a run survives it.
