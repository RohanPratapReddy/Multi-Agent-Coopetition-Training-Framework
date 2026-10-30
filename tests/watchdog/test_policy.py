"""test policy (split from test_watchdog.py)."""

from __future__ import annotations

from types import SimpleNamespace


import pytest


from mactf.exceptions import ImproperlyConfigured


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


@pytest.mark.parametrize(
    "bad",
    [
        {"nan_policy": "maybe"},
        {"inf_policy": "x"},
        {"loss_increase_ratio": 0.5},
        {"loss_patience": 0},
        {"max_grad_norm": 0},
        {"score_regression_tolerance": -1},
        {"entropy_collapse_ratio": 1.5},
        {"kappa_min": 0},
        {"kappa_min": 5, "kappa_max": 1},
        {"stale_epoch_seconds": 0},
        {"min_throughput_ratio": 0},
        {"max_step_time_drift": 0},
        {"max_rss_growth_ratio": 0.5},
        {"max_gpu_memory_ratio": 1.5},
        {"max_checkpoint_age_seconds": 0},
        {"max_tree_growth_ratio": 0.9},
        {"depth_cost_ratio": 1.0},
        {"warning_spike_threshold": 0},
    ],
)
def test_policy_rejects_bad_bounds(bad):
    with pytest.raises(ImproperlyConfigured):
        WatchdogPolicy(**bad)


def test_policy_from_settings_ignores_unknown_keys():
    policy = WatchdogPolicy.from_settings({"WATCHDOG": {"MAX_GRAD_NORM": 5.0, "OTHER": 1}})
    assert policy.max_grad_norm == 5.0


def test_unknown_action_rejected_at_construction():
    with pytest.raises(ImproperlyConfigured, match="ON_CRITICAL"):
        _dog(on_critical="explode")
    with pytest.raises(ImproperlyConfigured, match="ON_DEGRADATION"):
        _dog(on_degradation="explode")
    with pytest.raises(ImproperlyConfigured):
        MACTFWatchdog(checkpoint=None, logger=MACTFLogger(sinks=[], async_write=False))


def test_state_dict_roundtrip():
    dog, _, _ = _dog(snapshot_every_epochs=1)
    dog.tick(HealthContext(epoch=3))
    state = dog.state_dict()
    other, _, _ = _dog()
    other.load_state_dict(state)
    assert other._last_snapshot_epoch == 3
    assert other.describe()["on_critical"] == "rollback"
