# Observability

`MACTFLogger` (`mactf.training.logging`) writes typed records to one or more sinks. It is installed
with `mactf.contrib.telemetry` and configured under `TELEMETRY` in settings.

## Records

`EpochRecord`, `StepRecord`, `LossRecord`, `ScoreRecord`, `AggregationRecord`, `WarningRecord`,
`CheckpointRecord`, `HealthRecord`, `ResumeRecord`. Each carries `run_id`, a timestamp and a
monotonic sequence number, and serialises to JSON.

## Sinks

| Sink | Writes |
|---|---|
| `ConsoleSink` | human-readable lines to stderr |
| `FileSink` | text log with rotation |
| `JsonlSink` | one JSON record per line, for machines |
| `MemorySink` | records in a list; used by tests |
| `TensorBoardSink` | scalars (needs `tensorboard`) |
| `WandBSink` | metrics to Weights & Biases (needs `wandb`) |

TensorBoard and W&B are optional extras; selecting one without the package installed is an error
naming it, not a silent no-op. A sink that raises is counted and disabled; it cannot stop
training. `MACTFLogger(run_id=..., sinks=[...], async_write=False)` writes synchronously, which is
what tests and short scripts want.

## Over HTTP

| Route | Client call |
|---|---|
| `GET /runs/{id}/logs/tail` | `client.observability.logs(run_id, n=...)` |
| `GET /runs/{id}/watchdog` | `client.observability.health(run_id)` |
| `POST /runs/{id}/watchdog/snapshot` | `client.observability.snapshot(run_id)` |
| `GET /runs/{id}/events` | `client.runs.stream(run_id)` (server-sent events) |
| `POST /runs/{id}/resume` | `client.observability.resume(run_id, epochs=...)` |

Every error response carries a `trace_id`, also sent as `X-Trace-Id`; quote it when reporting a
problem. See [api.md](api.md) for the route list and [recovery.md](recovery.md) for the watchdog
whose status `health(run_id)` reports.
