"""Criterion 28: a three-level tree trains every node and aggregates per level."""

from __future__ import annotations

import pytest
import torch

from mactf.exceptions import ImproperlyConfigured
from mactf.coopetition.tree import HierarchySpec, LevelSpec

from hier_support import Tiny, three_level_spec, three_level_trainer, tuned  # noqa: F401


def test_every_node_gets_a_peer_and_an_optimizer(tuned, tmp_path):  # noqa: F811
    trainer = three_level_trainer(tuned, tmp_path).build()
    # 2 roots + 4 sub + 8 mini
    assert len(trainer.tree) == 14
    assert trainer.tree.depth == 3
    assert set(trainer.peers) == {node.path for node in trainer.tree.walk("root-first")}
    assert set(trainer.optimizers) == set(trainer.peers)


def test_every_node_takes_a_step_in_an_epoch(tuned, tmp_path):  # noqa: F811
    stepped: list[str] = []
    trainer = three_level_trainer(
        tuned, tmp_path, hooks={"on_step": lambda t, node, result: stepped.append(node.path)}
    )
    result = trainer.fit(epochs=1)
    assert result.status == "completed"
    assert len(stepped) == 14
    assert len(set(stepped)) == 14


def test_parameters_change_at_every_depth(tuned, tmp_path):  # noqa: F811
    trainer = three_level_trainer(tuned, tmp_path, peer_strategy="independent").build()
    before = {path: peer.export_parameters() for path, peer in trainer.peers.items()}
    trainer.fit(epochs=1)
    changed_depths = set()
    for path, peer in trainer.peers.items():
        if not torch.equal(peer.export_parameters(), before[path]):
            changed_depths.add(trainer.tree[path].depth)
    assert changed_depths == {1, 2, 3}


def test_each_level_is_aggregated_over_its_own_group(tuned, tmp_path):  # noqa: F811
    trainer = three_level_trainer(tuned, tmp_path)
    trainer.fit(epochs=1)
    groups = [tuple(report.weights) for report in trainer.reports]
    # one root group, two sub groups per root, one mini group per sub
    assert len(groups) == 7
    assert groups.count(("peer-1", "peer-2")) == 1
    assert ("peer-1/sub-1", "peer-1/sub-2") in groups
    assert ("peer-2/sub-1/mini-1", "peer-2/sub-1/mini-2") in groups
    for report in trainer.reports:
        assert sum(report.weights.values()) == pytest.approx(1.0, abs=1e-5)


def test_siblings_agree_after_their_group_aggregates(tuned, tmp_path):  # noqa: F811
    trainer = three_level_trainer(tuned, tmp_path)
    trainer.fit(epochs=1)
    first = trainer.peers["peer-1/sub-1/mini-1"].export_parameters()
    second = trainer.peers["peer-1/sub-1/mini-2"].export_parameters()
    assert torch.allclose(first, second)


def test_a_hierarchy_node_without_a_model_source_is_named(tuned, tmp_path):  # noqa: F811
    spec = HierarchySpec(
        levels=(LevelSpec("sub", 2),), roots=1, root_models={"peer-1": Tiny()}
    )
    trainer = three_level_trainer(tuned, tmp_path, spec=spec)
    with pytest.raises(ImproperlyConfigured, match=r"levels\[0\]\.model"):
        trainer.build()


def test_a_callable_model_source_receives_the_node_path(tuned, tmp_path):  # noqa: F811
    seen: list[str] = []

    def factory(path):
        seen.append(path)
        return Tiny()

    spec = HierarchySpec(
        levels=(LevelSpec("sub", 2, model=factory),),
        roots=2,
        root_models={"peer-1": factory, "peer-2": factory},
    )
    three_level_trainer(tuned, tmp_path, spec=spec).build()
    assert sorted(seen) == sorted(
        ["peer-1", "peer-2", "peer-1/sub-1", "peer-1/sub-2", "peer-2/sub-1", "peer-2/sub-2"]
    )


def test_the_epoch_is_deterministic_for_a_seed(tuned, tmp_path):  # noqa: F811
    def run(name):
        torch.manual_seed(5)
        trainer = three_level_trainer(tuned, tmp_path / name, seed=5, spec=three_level_spec(2, 2))
        trainer.fit(epochs=1)
        return trainer.peers["peer-1/sub-1/mini-1"].export_parameters()

    first, second = run("a"), run("b")
    assert torch.equal(first, second)
