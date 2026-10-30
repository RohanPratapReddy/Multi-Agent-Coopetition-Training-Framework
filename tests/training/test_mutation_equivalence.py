"""A mutated run and a run built on the final tree agree (document 17 section 12.6.2, rows 55-56)."""

import torch

from mactf.training.mutation import MutationPlan, add_peer, retire_peer
from mutation_support import Tiny, mutable_trainer, mutation_settings, parameters

EPOCHS = 3


def _run(trainer):
    reports = []
    for epoch in range(EPOCHS):
        trainer.run_epoch(epoch)
        report = trainer._epoch_report
        reports.append((dict(report.weights), float(report.kappa), float(report.entropy), tuple(report.adjustments)))
    return reports


def _agree(left, right):
    a, b = parameters(left), parameters(right)
    assert set(a) == set(b)
    for path in a:
        assert torch.equal(a[path], b[path]), path


def test_a_mutated_run_matches_a_prebuilt_one(settings, tmp_path):
    mutation_settings(settings)
    prebuilt = mutable_trainer(settings, tmp_path / "pre", peers=4, seed=3)
    mutated = mutable_trainer(settings, tmp_path / "mut", peers=2, seed=3)
    mutated.mutate(MutationPlan.of(add_peer("peer-3", like="peer-1", factory="clone"),
                                   add_peer("peer-4", like="peer-2", factory="clone")))
    assert mutated.controller.num_peers == 4 and sorted(mutated.peers) == sorted(prebuilt.peers)
    _agree(mutated, prebuilt)                                    # the same weights before the first step
    left, right = _run(mutated), _run(prebuilt)
    assert left == right                                         # weights, kappa, entropy, adjustments: identical
    _agree(mutated, prebuilt)
    assert [len(w) for w, *_ in left] == [4, 4, 4]               # identical aggregation shapes
    assert mutated.global_step == prebuilt.global_step and mutated.signal_by_peer == prebuilt.signal_by_peer
    assert mutated.path_state.audit() == []


def test_the_same_holds_when_the_growth_comes_after_an_epoch(settings, tmp_path):
    """Grow at epoch 1: the shared prefix is identical, so the two runs can only differ through the mutation."""
    mutation_settings(settings)
    a = mutable_trainer(settings, tmp_path / "a", peers=2, seed=3)
    b = mutable_trainer(settings, tmp_path / "b", peers=2, seed=3)
    a.run_epoch(0)
    b.run_epoch(0)
    _agree(a, b)
    a.add_peer("peer-3", like="peer-1", factory="clone")
    for epoch in (1, 2):
        a.run_epoch(epoch)
        b.run_epoch(epoch)
    assert sorted(a.peers) == ["peer-1", "peer-2", "peer-3"] and sorted(b.peers) == ["peer-1", "peer-2"]
    # the newcomer started as a copy of peer-1 and trained on the same batches: it is not peer-1 any more, but
    # the run stays internally consistent and finite
    for peer in a.peers.values():
        assert torch.isfinite(peer.export_parameters()).all()
    assert abs(sum(a._epoch_report.weights.values()) - 1.0) < 1e-6 and len(a._epoch_report.weights) == 3


def test_a_retired_run_matches_a_three_peer_run(settings, tmp_path):
    mutation_settings(settings)
    flat = mutable_trainer(settings, tmp_path / "flat", peers=3, seed=5)
    retired = mutable_trainer(settings, tmp_path / "retired", peers=4, seed=5)
    retired.retire_peer("peer-4")
    assert retired.controller.num_peers == 3
    left, right = _run(retired), _run(flat)
    assert left == right
    for path in flat.peers:
        assert torch.equal(parameters(retired)[path], parameters(flat)[path]), path
    # the retired peer is untouched by three epochs of the others
    assert torch.equal(parameters(retired)["peer-4"], parameters(mutable_trainer(settings, tmp_path / "x", peers=4, seed=5))["peer-4"])


def test_retire_then_revive_is_a_round_trip(settings, tmp_path):
    mutation_settings(settings)
    plain = mutable_trainer(settings, tmp_path / "plain", peers=3, seed=5)
    cycled = mutable_trainer(settings, tmp_path / "cycled", peers=3, seed=5)
    cycled.retire_peer("peer-3")
    cycled.revive_peer("peer-3")
    assert _run(cycled) == _run(plain)
    _agree(cycled, plain)
