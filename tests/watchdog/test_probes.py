"""test probes (split from test_watchdog.py)."""

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


def _probe(name, context, policy=None):
    probes = {p.name: p for p in default_probes(policy or WatchdogPolicy())}
    return probes[name].probe(context)


def test_nan_inf_loss():
    assert _probe("nan_loss", HealthContext(losses=[1.0, float("nan")])).status is C
    assert _probe("nan_loss", HealthContext(losses=[1.0])).status is H
    assert _probe("inf_loss", HealthContext(losses=[float("inf")])).status is C


def test_grad_norm_levels():
    pol = WatchdogPolicy(max_grad_norm=10.0)
    assert _probe("grad_norm", HealthContext(grad_norms=[1.0]), pol).status is H
    assert _probe("grad_norm", HealthContext(grad_norms=[50.0]), pol).status is D
    assert _probe("grad_norm", HealthContext(grad_norms=[500.0]), pol).status is C


def test_loss_trend_levels():
    assert _probe("loss_trend", HealthContext(losses=[1, 1, 1, 1.1])).status is H
    assert _probe("loss_trend", HealthContext(losses=[1, 1, 1, 4, 4])).status is D
    assert _probe("loss_trend", HealthContext(losses=[1, 1, 1, 50])).status is C


def test_score_trend_levels():
    assert _probe("score_trend", HealthContext(scores={"a": 0.9}, previous_scores={"a": 0.9})).status is H
    assert _probe("score_trend", HealthContext(scores={"a": 0.5}, previous_scores={"a": 0.9})).status is D
    assert _probe("score_trend", HealthContext(scores={"a": -1}, previous_scores={"a": 0.1})).status is C


def test_entropy_levels():
    assert _probe("entropy", HealthContext(entropy=0.69, max_entropy=0.69)).status is H
    assert _probe("entropy", HealthContext(entropy=0.01, max_entropy=0.69)).status is D
    assert _probe("entropy", HealthContext(entropy=0.0, max_entropy=0.69)).status is C
    assert _probe("entropy", HealthContext()).status is H


def test_aggregation_fallback_levels():
    ok, bad = SimpleNamespace(fallback=False), SimpleNamespace(fallback=True)
    assert _probe("aggregation_fallback", HealthContext(aggregation_reports=[ok])).status is H
    assert _probe("aggregation_fallback", HealthContext(aggregation_reports=[ok, bad])).status is D
    assert _probe("aggregation_fallback", HealthContext(aggregation_reports=[bad, bad])).status is C


def test_kappa_levels():
    assert _probe("kappa", HealthContext(kappa=1.0)).status is H
    assert _probe("kappa", HealthContext(kappa=1e6)).status is D
    assert _probe("kappa", HealthContext(kappa=float("nan"))).status is C


def test_stale_throughput_memory_age_spike():
    pol = WatchdogPolicy(stale_epoch_seconds=10)
    stale = lambda secs: HealthContext(now=1.0, monotonic=secs, extras={"epoch_started": 0.0})
    assert _probe("stale_progress", stale(5), pol).status is H
    assert _probe("stale_progress", stale(15), pol).status is D
    assert _probe("stale_progress", stale(25), pol).status is C

    thr = lambda cur: HealthContext(steps_per_second=cur, previous_steps_per_second=10.0)
    assert _probe("throughput", thr(9)).status is H
    assert _probe("throughput", thr(2)).status is D
    assert _probe("throughput", thr(0.5)).status is C

    assert _probe("memory", HealthContext(rss_bytes=100, initial_rss_bytes=100)).status is H
    assert _probe("memory", HealthContext(rss_bytes=300, initial_rss_bytes=100)).status is D
    assert _probe("memory", HealthContext(gpu_memory_ratio=0.99)).status is D

    assert _probe("checkpoint_age", HealthContext(epoch=1)).status is H
    assert _probe("checkpoint_age", HealthContext(epoch=5)).status is C
    old = HealthContext(epoch=5, checkpoint_count=1, last_checkpoint_at=0.0, now=99999.0)
    assert _probe("checkpoint_age", old).status is D

    assert _probe("warning_spike", HealthContext(warnings_escalated=1)).status is H
    assert _probe("warning_spike", HealthContext(warnings_escalated=11)).status is D
    assert _probe("warning_spike", HealthContext(warnings_escalated=25)).status is C


def test_hierarchy_probes():
    pol = WatchdogPolicy()
    dead = DeadBranchProbe(pol).probe(HealthContext(dead_branches={"p/a": 2}))
    assert dead.status is D and dead.node_path == "p/a"
    assert DeadBranchProbe(pol).probe(HealthContext(dead_branches={"p/a": 1})).status is H
    starved = ChildStarvationProbe(pol).probe(HealthContext(starved_children={"p/b": 3}))
    assert starved.status is D and starved.node_path == "p/b"
    cost = DepthCostProbe(pol).probe(HealthContext(level_durations={1: 1.0, 2: 10.0}))
    assert cost.status is D
    assert DepthCostProbe(pol).probe(HealthContext(level_durations={1: 1.0, 2: 2.0})).status is H


def test_raising_probe_becomes_critical_and_run_continues():
    class Broken:
        name = "broken"

        def probe(self, context):
            raise RuntimeError("bug")

    dog, ckpt, _ = _dog(probes=[Broken()], on_critical="log")
    report = dog.tick(HealthContext(epoch=1))
    assert report.status is C
    assert report.results[0].metadata["probe_error"] is True
    assert not dog.should_stop()
