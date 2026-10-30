"""The phase bridge and the signals the run manager listens to (document 17 section 13.7, rows 13-15)."""

from __future__ import annotations

import copy
from collections import Counter

import pytest

from engine_support import engine_of, fresh, tune

from mactf import signals
from mactf.orchestration.orchestrator import Orchestrator, RunConfig
from mactf.training.callbacks import Callback


class Tape:
    """Records the phase and episode signals of one run, in order."""

    SIGNALS = ("phase_started", "phase_finished", "episode_started", "episode_scored", "episode_peer_scored",
               "aggregation_completed", "checkpoint_written", "checkpoint_rolled_back")

    def __init__(self):
        self.events: list[tuple[str, str | None]] = []
        self._connected = []

    def __enter__(self):
        for name in self.SIGNALS:
            def receiver(signal=None, sender=None, __name=name, **payload):
                self.events.append((__name, payload.get("phase")))

            getattr(signals, name).connect(receiver, weak=False)
            self._connected.append((getattr(signals, name), receiver))
        return self

    def __exit__(self, *exc):
        for signal, receiver in self._connected:
            signal.disconnect(receiver)

    def phases(self):
        return [(kind.removeprefix("phase_"), phase) for kind, phase in self.events if kind.startswith("phase_")]

    def count(self, name):
        return sum(1 for kind, _ in self.events if kind == name)


def test_phase_order(settings, tmp_path):
    """The exact sequence ``bootstrap, debate, scoring, local_update, aggregation`` per epoch, as the orchestrator emits it."""
    tune(settings, tmp_path)
    with Tape() as oracle:
        Orchestrator(settings, modules=copy.deepcopy(fresh())).run(RunConfig(epochs=2, seed=1))
    with Tape() as tape:
        engine_of(settings, tmp_path).run(RunConfig(epochs=2, seed=1))
    assert tape.phases() == oracle.phases()
    names = [phase for kind, phase in tape.phases() if kind == "started"]
    assert names == ["bootstrap"] + ["debate", "scoring", "local_update", "aggregation"] * 2
    for phase in set(names):                                       # every start has its finish, in the same order
        assert [k for k, p in tape.phases() if p == phase] == ["started", "finished"] * names.count(phase)


def test_no_phase_left_open_on_failure(settings, tmp_path):
    """T7: an epoch that raises mid-update still closes every phase it opened."""

    class Explodes(Callback):
        name = "explodes"

        def on_score(self, trainer, card):
            raise RuntimeError("the update cannot start")

    tune(settings, tmp_path)
    engine = engine_of(settings, tmp_path)
    config = RunConfig(epochs=2)
    engine.prepare(config)
    engine.trainer.add_callback(Explodes())
    with Tape() as tape:
        with pytest.raises(RuntimeError, match="cannot start"):
            engine.run(config)
    counts = Counter(tape.phases())
    for phase in ("debate", "scoring", "local_update"):
        assert counts[("started", phase)] == counts[("finished", phase)] == 1, phase
    assert engine._bridge.open == []
    assert engine.registry.get_run(engine.run_id)["status"] == "failed"


def test_episode_and_aggregation_signals_fire_once(settings, tmp_path):
    tune(settings, tmp_path, epochs=3)
    with Tape() as oracle:
        Orchestrator(settings, modules=copy.deepcopy(fresh())).run(RunConfig(epochs=3, seed=1))
    with Tape() as tape:
        engine_of(settings, tmp_path).run(RunConfig(epochs=3, seed=1))
    for name in ("episode_scored", "aggregation_completed", "checkpoint_written"):
        assert tape.count(name) == oracle.count(name) == 3, name
    assert tape.count("episode_peer_scored") == oracle.count("episode_peer_scored") == 6
    assert tape.count("checkpoint_rolled_back") == 0
    # D-13.10: the trainer rolls each node out on its own (``EpisodeRunner.run`` per node), so
    # ``episode_started`` fires once per node and epoch where the orchestrator's single call fires once
    # per epoch. Nothing the run manager subscribes to reads it.
    assert oracle.count("episode_started") == 3 and tape.count("episode_started") == 3 * 2


def test_a_strategy_that_aggregates_nothing_still_closes_its_phases(settings, tmp_path):
    """``independent`` never dispatches ``on_aggregate``; ``on_epoch_end`` closes the update."""
    tune(settings, tmp_path)
    settings["TRAINER"]["PEER_STRATEGY"] = "independent"
    with Tape() as tape:
        summary = engine_of(settings, tmp_path).run(RunConfig(epochs=2, seed=1))
    assert summary["status"] == "completed"
    counts = Counter(tape.phases())
    for phase in ("debate", "scoring", "local_update", "aggregation"):
        assert counts[("started", phase)] == counts[("finished", phase)], phase     # never a phase left open
    assert counts[("started", "local_update")] == 2
