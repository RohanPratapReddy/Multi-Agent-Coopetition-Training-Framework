"""Real bootstrap: behavioural cloning with measurable loss movement."""

from __future__ import annotations

import torch

from mactf.data.sources import SyntheticTaskSource
from mactf.orchestration.bootstrap import BootstrapConfig, BootstrapTrainer
from mactf.orchestration.episode import Task
from mactf.orchestration.phases import PhaseContext, bootstrap
from mactf.coopetition.peer import build_peer, build_reference
from mactf.coopetition.aggregation import Aggregator
from mactf.runtime import build_runtime
from mactf.scoring.pipeline import build_scorer


class _Learner(torch.nn.Module):
    """A tiny module whose target rule is learnable by construction."""

    def __init__(self, state_dim: int = 8, action_dim: int = 4) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(state_dim, action_dim)
        self.value = torch.nn.Linear(state_dim, 1)

    def forward(self, states: torch.Tensor):
        return self.linear(states), self.value(states)


def _peers(runtime, count: int = 2, seed: int = 0):
    peers = {}
    for index in range(count):
        torch.manual_seed(seed + index)
        name = f"peer_{index}"
        peers[name] = build_peer(
            _Learner(), name=name, runtime=runtime, learning_rate=1e-3
        )
    return peers


def _runtime():
    from mactf.conf import Settings
    from mactf.conf import DEFAULTS

    import copy

    values = copy.deepcopy(DEFAULTS)
    values["DEVICE"] = "cpu"
    values["DTYPE"] = "float32"
    values["AMP"] = False
    values["SEED"] = 3
    return build_runtime(Settings(values, module="test"))


def _trainer(**kwargs):
    runtime = _runtime()
    source = SyntheticTaskSource(state_dim=8, action_dim=4, batch_size=16, seed=5)
    return runtime, source, BootstrapTrainer(
        source=source, runtime=runtime, config=BootstrapConfig(**kwargs)
    )


def _orchestrator(project_settings):
    from mactf import apps as apps_module
    from mactf.orchestration.orchestrator import Orchestrator

    apps_module.reset_registry()
    apps_module.populate(list(project_settings["INSTALLED_APPS"]))
    return Orchestrator(project_settings)


def test_loss_decreases_and_report_is_complete():
    runtime, source, trainer = _trainer(epochs=3, batches_per_epoch=4, learning_rate=5e-3)
    peers = _peers(runtime)
    reference = build_reference(next(iter(peers.values())), runtime=runtime)
    report = trainer.train(peers, reference)

    assert report.epochs_run == 3
    assert report.batches == 12
    assert report.examples == 12 * 16
    assert report.final_loss < report.initial_loss, report.loss_history
    assert set(report.peers) == set(peers)
    assert report.reference_updated is True
    assert "bootstrap(" in report.describe()


def test_reference_changes_only_when_requested():
    for update in (True, False):
        runtime, source, trainer = _trainer(
            epochs=1, batches_per_epoch=2, update_reference=update
        )
        peers = _peers(runtime)
        reference = build_reference(next(iter(peers.values())), runtime=runtime)
        before = reference.export_parameters().clone()
        report = trainer.train(peers, reference)
        changed = not torch.equal(before, reference.export_parameters())
        assert report.reference_updated is update
        assert changed is update


def test_zero_epochs_does_nothing():
    runtime, source, trainer = _trainer(epochs=0)
    peers = _peers(runtime)
    before = [peer.export_parameters().clone() for peer in peers.values()]
    report = trainer.train(peers, None)
    assert report.epochs_run == 0
    assert report.examples == 0
    for original, peer in zip(before, peers.values()):
        assert torch.equal(original, peer.export_parameters())


def test_patience_stops_early():
    runtime, source, trainer = _trainer(epochs=6, batches_per_epoch=1, patience=1, learning_rate=0.0)
    peers = _peers(runtime)
    report = trainer.train(peers, None)
    assert report.stopped_early is True
    assert report.epochs_run < 6
    assert report.stop_reason is not None


def test_bootstrap_phase_disabled_leaves_parameters_identical(project_settings):
    orchestrator = _orchestrator(project_settings)
    context = orchestrator.prepare()
    before = [peer.export_parameters().clone() for peer in context.peers.values()]
    summary = bootstrap(context, enabled=False)
    assert summary["epochs_run"] == 0
    for original, peer in zip(before, context.peers.values()):
        assert torch.equal(original, peer.export_parameters())


def test_bootstrap_phase_reports_real_loss(project_settings):
    project_settings["TRAINING"]["BOOTSTRAP_CONFIG"]["EPOCHS"] = 2
    project_settings["TRAINING"]["BOOTSTRAP_CONFIG"]["BATCHES_PER_EPOCH"] = 2
    orchestrator = _orchestrator(project_settings)
    context = orchestrator.prepare()
    summary = bootstrap(context, enabled=True)
    assert summary["enabled"] is True
    assert summary["epochs_run"] == 2
    assert len(summary["loss_history"]) == 2
    assert summary["examples"] > 0
    assert "bootstrap" in context.metrics


def test_bootstrap_emits_no_warning_signals(project_settings):
    from mactf import signals

    project_settings["TRAINING"]["BOOTSTRAP_CONFIG"]["EPOCHS"] = 2
    issued: list = []
    receiver = lambda **kwargs: issued.append(kwargs)  # noqa: E731
    signals.warning_issued.connect(receiver, weak=False)
    try:
        orchestrator = _orchestrator(project_settings)
        context = orchestrator.prepare()
        bootstrap(context, enabled=True)
    finally:
        signals.warning_issued.disconnect(receiver)
    assert issued == []
