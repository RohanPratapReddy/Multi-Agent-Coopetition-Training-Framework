"""test health log (split from test_watchdog.py)."""

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


def test_health_is_logged_and_history_is_bounded():
    dog, _, log = _dog(on_critical="log", keep_history=3)
    for epoch in range(1, 6):
        dog.tick(HealthContext(epoch=epoch, kappa=float("nan")))
    assert len(dog.health_history(0)) == 3
    assert len(log.tail(0, kind="health")) == 5
    assert log.tail(1, kind="health")[0].level == "error"


def test_watchdog_never_mutates_the_run():
    import torch

    module = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    before = [p.detach().clone() for p in module.parameters()]
    ctx = HealthContext(epoch=2, losses=[1.0, float("nan")], extras={"module": module})
    dog, _, _ = _dog(on_critical="log")
    dog.tick(ctx)
    assert optimizer.param_groups[0]["lr"] == 0.1
    assert all(torch.equal(a, b) for a, b in zip(before, module.parameters()))
