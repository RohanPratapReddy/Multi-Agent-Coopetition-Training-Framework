"""Performance sanity of the default path, and a simple leak measurement over a long run.

Timings are medians of several repeats, compared with a *generous* bound: the point is to catch an order-of-
magnitude regression (a registry call per step, a snapshot per episode), not to benchmark. The measured numbers
are printed with ``-s`` and recorded in the implementation notes.
"""

from __future__ import annotations

import gc
import logging
import os
import statistics
import time
import tracemalloc

import pytest

from component_helpers import Policy
from engine_support import engine_of, tune
from mactf.components import HUB, Hub

SLOW = pytest.mark.skipif(os.environ.get("MACTF_E2E_SKIP_SLOW") == "1", reason="slow")


def _time_engine(settings, tmp_path, engine, epochs, tag):
    tune(settings, tmp_path / tag, epochs=epochs)
    runner = engine_of(settings, tmp_path, engine=engine)
    start = time.perf_counter()
    runner.run()
    elapsed = time.perf_counter() - start
    close = getattr(runner, "close", None)
    if callable(close):
        close()
    return elapsed


def test_the_default_trainer_engine_is_not_slower_than_the_orchestrator_by_a_wide_margin(settings, tmp_path):
    epochs = 6
    times = {"orchestrator": [], "trainer": []}
    for repeat in range(3):
        for engine in times:                       # interleave, so drift hits both
            times[engine].append(_time_engine(settings, tmp_path, engine, epochs, f"{engine}{repeat}"))
    median = {engine: statistics.median(values) for engine, values in times.items()}
    ratio = median["trainer"] / median["orchestrator"]
    print(f"\nPERF per run of {epochs} epochs: orchestrator {median['orchestrator']:.3f}s, trainer {median['trainer']:.3f}s, ratio {ratio:.2f}")
    assert ratio < 4.0, median


def test_registry_reads_are_microseconds():
    hub = Hub()
    registry = hub.registry("trainer_policy")
    registry.register("p", Policy)
    count = 20000
    start = time.perf_counter()
    for _ in range(count):
        registry.find("p")
    per_find = (time.perf_counter() - start) / count
    with hub.scoped("run"):
        start = time.perf_counter()
        for _ in range(count):
            registry.find("p")
        per_scoped = (time.perf_counter() - start) / count
    print(f"\nPERF registry.find: {per_find * 1e6:.2f} us, inside an overlay {per_scoped * 1e6:.2f} us")
    assert per_find < 50e-6 and per_scoped < 100e-6


def _drop_record_keeping_handlers() -> None:
    """Measure the run, not the log records pytest's capture handlers keep alive.

    ``configure_logging`` (run by ``prepare``) attaches the root handlers, pytest's capture handlers among them, to
    the non-propagating ``mactf`` logger; those hold every record, and the tensors in its arguments, for the session.
    """
    for name in ("", "mactf"):
        logger = logging.getLogger(name)
        logger.handlers[:] = [h for h in logger.handlers if not hasattr(h, "records")]


@SLOW
def test_a_long_run_does_not_leak_memory_or_handles(settings, tmp_path):
    tune(settings, tmp_path / "long", epochs=60)
    settings["CHECKPOINT"]["KEEP"] = 2
    runner = engine_of(settings, tmp_path, engine="trainer")
    context = runner.prepare()
    del context
    _drop_record_keeping_handlers()
    fds_before = len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else None
    gc.collect()
    tracemalloc.start()
    first = None
    marks = []
    trainer = runner.trainer
    for epoch in range(60):
        trainer.run_epoch(epoch)
        if epoch in (9, 59):
            gc.collect()
            marks.append(tracemalloc.get_traced_memory()[0])
        if epoch == 9:
            first = marks[0]
    tracemalloc.stop()
    growth = marks[1] - first
    fds_after = len(os.listdir("/proc/self/fd")) if fds_before is not None else None
    print(f"\nPERF memory between epoch 10 and 60: {growth / 1024:.1f} KiB; fds {fds_before} -> {fds_after}")
    assert growth < 8 * 1024 * 1024, f"{growth} bytes retained over 50 epochs"
    if fds_before is not None:
        assert fds_after - fds_before <= 3
    runner.close()
    gc.collect()
    if fds_before is not None:
        assert len(os.listdir("/proc/self/fd")) - fds_before <= 3
