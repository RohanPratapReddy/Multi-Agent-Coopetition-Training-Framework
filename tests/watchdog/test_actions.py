"""test actions (split from test_watchdog.py)."""

from __future__ import annotations

from types import SimpleNamespace


from mactf.training.logging import MACTFLogger


from mactf.training.watchdog import (
    ChildStarvationProbe,
    DeadBranchProbe,
    DepthCostProbe,
    HealthContext,
    MACTFWatchdog,
    WatchdogAction,
    WatchdogPolicy,
    WatchdogStatus,
    default_probes,
)


H, D, C = WatchdogStatus.HEALTHY, WatchdogStatus.DEGRADED, WatchdogStatus.CRITICAL


class FakeCheckpoint:
    def __init__(self):
        self.calls: list[tuple] = []

    def save(self, context, *, reason, episode, best=False):
        self.calls.append(("save", reason, episode))
        return SimpleNamespace(path="x", episode=episode)

    def resume_from_last_checkpoint(self):
        self.calls.append(("resume",))
        return "resumed"


def _dog(policy=None, **kwargs):
    ckpt = FakeCheckpoint()
    log = MACTFLogger(run_id="w", sinks=[], async_write=False)
    dog = MACTFWatchdog(checkpoint=ckpt, logger=log, policy=policy, **kwargs)
    return dog, ckpt, log


def test_snapshot_precedes_rollback():
    dog, ckpt, _ = _dog(on_critical="rollback")
    report = dog.tick(HealthContext(epoch=2, losses=[1.0, float("nan")]))
    assert report.action is WatchdogAction.ROLLBACK
    assert [c[0] for c in ckpt.calls] == ["save", "resume"]
    assert report.snapshot is not None


def test_pause_and_abort_stop_training_after_snapshot():
    for action in ("pause", "abort"):
        dog, ckpt, _ = _dog(on_critical=action)
        dog.tick(HealthContext(epoch=2, losses=[float("nan")]))
        assert dog.should_stop()
        assert ckpt.calls[0][0] == "save"


def test_periodic_snapshot_and_degradation_policy():
    dog, ckpt, _ = _dog(snapshot_every_epochs=2)
    assert dog.tick(HealthContext(epoch=1)).action is WatchdogAction.NONE
    assert dog.tick(HealthContext(epoch=2, checkpoint_count=1)).action is WatchdogAction.SNAPSHOT
    assert ckpt.calls == [("save", "periodic", 2)]
    dog2, ckpt2, _ = _dog(on_degradation="log", snapshot_on_degradation=False)
    report = dog2.tick(HealthContext(epoch=1, kappa=1e9))
    assert report.status is D and report.action is WatchdogAction.LOG
    assert ckpt2.calls == []


def test_prune_branch_uses_tree():
    removed = []
    tree = SimpleNamespace(remove_node=lambda path, recursive: removed.append(path) or 1)
    dog, _, _ = _dog(on_critical="prune_branch", probes=[DeadBranchProbe(WatchdogPolicy())])
    ctx = HealthContext(epoch=1, dead_branches={"p/a": 5}, extras={"tree": tree})
    assert dog.prune_branch(ctx) == 1 and removed == ["p/a"]
