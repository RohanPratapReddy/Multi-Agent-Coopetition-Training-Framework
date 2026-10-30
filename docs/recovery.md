# Recovery

Two tools: `MACTFCheckpoint` (write and verify state, resume) and `MACTFWatchdog` (decide when to
use it). Both live in `mactf.orchestration.checkpoint_v2` and `mactf.training.watchdog`.

## Checkpoints

```python
from mactf.orchestration.checkpoint_v2 import MACTFCheckpoint

checkpoints = MACTFCheckpoint("runs/ckpt", run_id="demo")
```

A checkpoint is written atomically (temp file then rename) with a content hash, and records
sections: model/adapters, optimizers, schedulers, grad scaler, memory, rng, data position, warnings,
logging and watchdog state, plus the topology, the contract version and `SIGNAL_SOURCE`.
`verify` re-hashes a file and reports each section; `POST /runs/{id}/checkpoints/{n}/verify` and
`client.observability.verify_checkpoint` do the same remotely.

## Resume

`checkpoints.resume_from_last_checkpoint(trainer=trainer)` returns a `ResumeReport`.
`bit_identical_expected` is true when rng, data position, memory and optimizer state are all
present and the checkpoint verified strictly. Other absent sections (adapters, schedulers, grad
scaler, warnings, logging, watchdog) are listed in the report's notes but do not veto it, because
they do not change the numbers the next step computes.

Resume refuses, with a `ContractMismatch` naming the difference, when the contract version or the
tree topology differs, or when the checkpoint was written under a different `SIGNAL_SOURCE`.
Everything else it cannot restore is a warning in the report, never silent.

`POST /runs/{id}/resume` starts a new run from a stored checkpoint; `POST /runs/{id}/rollback` (or `.../checkpoints/{n}/rollback`)
re-points a run at an earlier episode.

## Watchdog

`MACTFWatchdog.tick(context)` evaluates loss, gradient, score, compliance and resource thresholds
(`WatchdogPolicy`, every bound validated at construction so a typo cannot silently disable it) and
returns a status: `healthy`, `degraded` or `critical`. A policy maps each status to an action:
`none`, `log`, `snapshot`, `rollback`, `prune_branch`, `pause`, `abort`; `pause` and `abort` stop
training. When a watchdog has a checkpoint, the trainer reuses it for periodic saves; a rollback
action restores the last good checkpoint and the run continues from there.

A NaN loss defaults to `abort` (`nan_policy`) because continuing from NaN weights corrupts every
later checkpoint.
