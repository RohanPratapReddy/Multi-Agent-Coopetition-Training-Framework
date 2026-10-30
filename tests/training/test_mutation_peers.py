"""Building a peer: factories, seeds, weights, layouts, ties (document 17 section 12.3, rows 31-37)."""

import copy

import pytest
import torch

from mactf.components import register_peer_factory
from mactf.exceptions import MutationError
from mactf.training.mutation import MutationPlan, add_peer
from mutation_support import Tiny, Wide, mutable_trainer, mutation_settings, parameters, snapshot


@pytest.fixture()
def trainer(settings, tmp_path):
    mutation_settings(settings)
    return mutable_trainer(settings, tmp_path, peers=3)


def _flat(peer):
    return peer.export_parameters().detach().clone()


def test_a_new_peer_is_seeded_from_the_run_seed(settings, tmp_path):
    """Same plan, same seed, two trainers: identical parameters; the run's own RNG is untouched (M4)."""
    mutation_settings(settings)
    runs = []
    for index in range(2):
        t = mutable_trainer(settings, tmp_path / str(index), peers=3, seed=11)
        torch.manual_seed(5)
        before = torch.get_rng_state().clone()
        t.add_peer("peer-4", like="peer-1", factory={"type": "perturbed_clone", "params": {"std": 0.05}})
        assert torch.equal(torch.get_rng_state(), before)
        runs.append(t)
    a, b = runs
    assert torch.equal(_flat(a.peers["peer-4"]), _flat(b.peers["peer-4"]))
    noise = _flat(a.peers["peer-4"]) - _flat(a.peers["peer-1"])
    assert noise.abs().max() > 0 and noise.abs().max() < 1.0                      # perturbed, not copied
    # a different path or a different topology epoch is a different stream
    a.add_peer("peer-5", like="peer-1", factory={"type": "perturbed_clone", "params": {"std": 0.05}})
    assert not torch.equal(_flat(a.peers["peer-5"]), _flat(a.peers["peer-4"]))
    # the plain `clone` is exactly the sibling
    a.add_peer("peer-6", like="peer-1", factory="clone")
    assert torch.equal(_flat(a.peers["peer-6"]), _flat(a.peers["peer-1"]))


def test_without_like_the_build_time_source_makes_the_module(trainer):
    trainer.add_peer("peer-4")                                       # the single-model source: a deep copy of `Tiny`
    assert trainer.peers["peer-4"].adapter.parameter_shapes() == trainer.peers["peer-1"].adapter.parameter_shapes()
    assert trainer.peers["peer-4"].metadata["depth"] == 1 and trainer.peers["peer-4"].metadata["role"] == "root"
    assert trainer.tree["peer-4"].peer is trainer.peers["peer-4"]


def test_init_from_copies_weights_strictly(trainer, component_overlay):
    trainer.add_peer("peer-4", like="peer-1", factory="perturbed_clone", init_from="peer-2")
    assert torch.equal(_flat(trainer.peers["peer-4"]), _flat(trainer.peers["peer-2"]))
    trainer.add_peer("peer-5", init_from="@reference")
    assert torch.equal(_flat(trainer.peers["peer-5"]), _flat(trainer.reference))

    @register_peer_factory("wide", scope="run", capabilities={"finetuned": True})
    def wide(spec, *, hidden: int = 8):
        return Wide(hidden)

    before = snapshot(trainer)
    with pytest.raises(MutationError) as caught:                       # a shape mismatch is E085
        trainer.add_peer("peer-6", factory="wide", init_from="peer-1")
    assert caught.value.code == "E085" and "layout" in str(caught.value)
    assert snapshot(trainer) == before
    # modify ... init_from re-seeds a live peer, with the same strictness
    trainer.modify_peer("peer-3", init_from="peer-1")
    assert torch.equal(_flat(trainer.peers["peer-3"]), _flat(trainer.peers["peer-1"]))
    with pytest.raises(MutationError):
        trainer.modify_peer("peer-3", init_from="ghost")


def test_a_wider_sibling_in_one_group_is_refused(trainer, component_overlay):
    @register_peer_factory("wide", scope="run", capabilities={"finetuned": True})
    def wide(spec, *, hidden: int = 8):
        return Wide(hidden)

    before = snapshot(trainer)
    with pytest.raises(MutationError) as caught:
        trainer.add_peer("peer-4", factory="wide")
    assert caught.value.code == "E085" and "'root'" in str(caught.value)
    assert snapshot(trainer) == before and trainer.path_state.audit() == []
    # a different layout is fine in a *different* group: two wide children under peer-1
    trainer.mutate(MutationPlan.of(
        add_peer("peer-1/a", factory="wide"), add_peer("peer-1/b", factory={"type": "wide", "params": {"hidden": 8}})
    ))
    assert {"peer-1/a", "peer-1/b"} <= set(trainer.peers)
    trainer.run_epoch(0)
    assert trainer.path_state.audit() == []
    # ...but not a Tiny sibling of those wide ones
    with pytest.raises(MutationError) as caught:
        trainer.add_peer("peer-1/c", like="peer-2")
    assert caught.value.code == "E085" and "peer-1" in str(caught.value)


def test_a_shared_trunk_is_not_unshared_by_like(settings, tmp_path):
    from mactf.coopetition.tree import HierarchySpec, LevelSpec
    from mactf.training import MACTFTrainer

    mutation_settings(settings)
    spec = HierarchySpec(
        levels=(LevelSpec("sub", 2, model=Tiny()),), roots=2, sharing="shared_trunk",
        root_models={"peer-1": Tiny(), "peer-2": Tiny()},
    )
    torch.manual_seed(0)
    trainer = MACTFTrainer(hierarchy=spec, settings=settings, checkpoint_dir=tmp_path / "ck").build()
    assert trainer._shared_trunk is not None
    before = snapshot(trainer)
    with pytest.raises(MutationError) as caught:
        trainer.mutate(MutationPlan.of(add_peer("peer-1/sub-9", like="peer-1/sub-1")))
    assert caught.value.code == "E085" and "unshare" in str(caught.value)
    assert snapshot(trainer) == before
    # the level's own model source is the way to add a child: it joins the shared trunk
    trainer.mutate(MutationPlan.of(add_peer("peer-1/sub-9")))
    trunk = {id(p) for p in trainer._shared_trunk.parameters()}
    shared = {id(p) for p in trainer.peers["peer-1/sub-9"].module.parameters()} & trunk
    assert shared and trainer.path_state.audit() == []
    trainer.run_epoch(0)


def test_tied_parameters_survive_the_tree(trainer, component_overlay):
    live = trainer.peers["peer-1"].module

    @register_peer_factory("tied", scope="run", capabilities={"finetuned": True})
    def tied(spec):
        return live                       # the very module of a sibling: every parameter is shared

    before = snapshot(trainer)
    with pytest.raises(MutationError) as caught:
        trainer.add_peer("peer-4", factory="tied")
    assert caught.value.code == "E085" and "tied" in str(caught.value)
    assert snapshot(trainer) == before

    @register_peer_factory("half_tied", scope="run", capabilities={"finetuned": True})
    def half_tied(spec):
        module = Tiny()
        module.policy = live.policy      # one tied layer is enough
        return module

    with pytest.raises(MutationError) as caught:
        trainer.add_peer("peer-4", factory="half_tied")
    assert caught.value.code == "E085" and "tied" in str(caught.value)
    # tied into a *different* group is refused as well (the cross-group rule)
    with pytest.raises(MutationError) as caught:
        trainer.mutate(MutationPlan.of(add_peer("peer-2/a", factory="half_tied"), add_peer("peer-2/b")))
    assert caught.value.code == "E085" and "tied" in str(caught.value)
    assert snapshot(trainer) == before


def test_a_retired_peer_does_not_hide_a_tie(trainer, component_overlay):
    """M7 is over *active* peers: a tie with a retired peer is not a tie in play."""
    live = trainer.peers["peer-3"].module

    @register_peer_factory("tied", scope="run", capabilities={"finetuned": True})
    def tied(spec):
        return live

    trainer.retire_peer("peer-3")
    trainer.add_peer("peer-4", factory="tied")
    assert trainer.path_state.audit() == []


def test_rebuild_resets_the_optimizer_and_warns(trainer):
    trainer.run_epoch(0)
    old = trainer.optimizers["peer-2"]
    old_peer = trainer.peers["peer-2"]
    report = trainer.modify_peer("peer-2", rebuild={"factory": "@build"})
    assert report.rebuilt == ("peer-2",) and report.optimizers_rebuilt == ("peer-2",)
    assert any(w.startswith("W043") and "peer-2" in w for w in report.warnings)
    assert trainer.optimizers["peer-2"] is not old and trainer.peers["peer-2"] is not old_peer
    assert trainer.peers["peer-2"].optimizer is trainer.optimizers["peer-2"]
    assert trainer.tree["peer-2"].peer is trainer.peers["peer-2"]
    # a fresh facade has taken no step and is bound to the *new* module's parameters
    assert not trainer.optimizers["peer-2"].state_dict()["state"]["state"]
    bound = {id(p) for p in trainer.peers["peer-2"].module.parameters()}
    assert {id(p) for p in trainer.optimizers["peer-2"].parameters} == bound
    assert trainer._node_specs["peer-2"]["factory"] == "@build"
    trainer.run_epoch(1)


def test_a_missing_factory_is_e085_with_the_name(trainer):
    before = snapshot(trainer)
    with pytest.raises(MutationError) as caught:
        trainer.add_peer("peer-4", like="peer-1", factory="no_such_factory")
    assert caught.value.code == "E085" and "no_such_factory" in str(caught.value) and "peer-4" in str(caught.value)
    assert snapshot(trainer) == before


def test_a_factory_that_returns_something_else_is_e085(trainer, component_overlay):
    @register_peer_factory("not_a_module", scope="run")
    def not_a_module(spec):
        return torch.nn.Linear(1, 1)           # a module, but see below: the contract is checked for non-modules

    @register_peer_factory("none_factory", scope="run")
    def none_factory(spec):
        return None

    with pytest.raises(MutationError) as caught:
        trainer.add_peer("peer-4", factory="none_factory")
    assert caught.value.code == "E085"


def test_from_state_loads_a_state_file(trainer, tmp_path):
    path = tmp_path / "state.pt"
    donor = copy.deepcopy(trainer.peers["peer-2"].module.state_dict())
    torch.save(donor, path)
    trainer.add_peer("peer-4", like="peer-1", factory={"type": "from_state", "params": {"path": str(path)}})
    assert torch.equal(_flat(trainer.peers["peer-4"]), _flat(trainer.peers["peer-2"]))
    with pytest.raises(MutationError, match="no state file"):
        trainer.add_peer("peer-5", like="peer-1", factory={"type": "from_state", "params": {"path": str(tmp_path / "x.pt")}})
    torch.save({"unrelated.weight": torch.zeros(1)}, path)
    with pytest.raises(MutationError, match="does not match"):
        trainer.add_peer("peer-5", like="peer-1", factory={"type": "from_state", "params": {"path": str(path)}})


def test_the_builtin_factories_are_registered_with_the_finetuned_capability():
    from mactf.components.registry import HUB

    registry = HUB.registry("peer_factory")
    for name in ("clone", "perturbed_clone", "from_state"):
        component = registry.get(name)
        assert component.origin.value == "builtin" and component.capabilities["finetuned"] is True
    with pytest.raises(ValueError, match="needs `like`"):
        registry.get("clone").factory(None, None)
    with pytest.raises(ValueError, match="std"):
        registry.get("perturbed_clone").factory(None, Tiny(), std=-1)
