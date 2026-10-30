"""Peer strategies: traits, dispatch, the five built-ins through the context, and the freeze loop.

Invariants T1 (default identity), T7 (frozen means frozen under strict), T11 (restore) and T12
(no dead settings) of document 17 section 11.2, and the four behaviour changes of 11.12.
"""

from __future__ import annotations

import copy
import logging

import pytest
import torch

from mactf.components import register_peer_strategy
from mactf.components.strategy_plan import DEFAULT_PATH, StrategyPlan, default_authored
from mactf.coopetition.strategy import (
    BUILTIN_STRATEGY_TYPES,
    PEER_STRATEGIES,
    CooperativeStrategy,
    FrozenPeerStrategy,
    GroupRef,
    HierarchicalStrategy,
    IndependentStrategy,
    KappaSoftmaxStrategy,
    PeerStrategy,
    StrategyTraits,
    TreeView,
    build_peer_strategy,
    coordinates_via_context,
    defines_nothing,
    traits_of,
    tree_view,
)
from mactf.coopetition.tree import PeerTree
from mactf.coopetition.weights import make_report
from mactf.exceptions import (
    ComponentParamError,
    ContractError,
    ImproperlyConfigured,
    ValidationError,
)
from mactf.training.freeze import FreezeSet

from p3_examples import StaggeredMixStrategy
from p3_support import flat, flat_parameters, hier, run_trace


# -- traits and dispatch ----------------------------------------------------------------------
def test_the_builtins_declare_their_traits() -> None:
    assert traits_of(IndependentStrategy()) == StrategyTraits(aggregates=False, walk="root")
    assert traits_of(CooperativeStrategy()).aggregates is False
    assert traits_of(KappaSoftmaxStrategy()).aggregates is True
    frozen = traits_of(FrozenPeerStrategy())
    assert frozen.aggregates and frozen.freezes and frozen.walk == "root"
    hierarchical = traits_of(HierarchicalStrategy())
    assert hierarchical.walk == "tree" and hierarchical.precompute_root and hierarchical.aggregates
    assert BUILTIN_STRATEGY_TYPES == frozenset(dict.values(PEER_STRATEGIES))
    assert HierarchicalStrategy().should_aggregate() and not IndependentStrategy().should_aggregate()


def test_a_bad_walk_is_refused() -> None:
    with pytest.raises(ValidationError, match="walk"):
        StrategyTraits(walk="sideways")


def test_traits_of_a_structural_strategy_come_from_should_aggregate() -> None:
    class Structural:
        def on_epoch_end(self, *a):
            pass

        def should_aggregate(self):
            return True

    class Broken(Structural):
        def should_aggregate(self):
            raise RuntimeError("cannot say")

    assert traits_of(Structural()).aggregates is True
    assert traits_of(Broken()).aggregates is False
    assert traits_of(object()).aggregates is False


def test_which_entry_point_runs() -> None:
    class Old(PeerStrategy):
        name = "old"

        def on_epoch_end(self, trainer, epoch, report):
            pass

    class New(PeerStrategy):
        name = "new"

        def coordinate(self, ctx):
            pass

    class Both(PeerStrategy):
        name = "both"

        def coordinate(self, ctx):
            pass

        def on_epoch_end(self, trainer, epoch, report):
            pass

    class LegacySubclass(HierarchicalStrategy):            # overrides only on_epoch_end: still honoured
        def on_epoch_end(self, trainer, epoch, report):
            super().on_epoch_end(trainer, epoch, report)

    class ContextSubclass(HierarchicalStrategy):           # overrides only coordinate
        def coordinate(self, ctx):
            super().coordinate(ctx)

    class Nothing(PeerStrategy):
        name = "nothing"

    assert not coordinates_via_context(Old()) and coordinates_via_context(New())
    assert coordinates_via_context(Both())
    assert not coordinates_via_context(LegacySubclass()) and coordinates_via_context(ContextSubclass())
    assert not coordinates_via_context(object())
    for builtin in BUILTIN_STRATEGY_TYPES:
        assert coordinates_via_context(builtin())
    assert defines_nothing(Nothing()) and not defines_nothing(Old()) and not defines_nothing(New())
    assert not defines_nothing(object())


def test_on_epoch_end_is_no_longer_abstract() -> None:
    class JustContext(PeerStrategy):
        name = "just_context"

        def coordinate(self, ctx):
            pass

    JustContext()                                     # does not raise TypeError
    assert JustContext().should_aggregate() is False


# -- build_peer_strategy over the resolver ------------------------------------------------------------
def test_build_peer_strategy_keeps_its_documented_forms() -> None:
    assert isinstance(build_peer_strategy(None, hierarchy=True), HierarchicalStrategy)
    assert isinstance(build_peer_strategy(None), IndependentStrategy)
    assert isinstance(build_peer_strategy("Hierarchical "), HierarchicalStrategy)
    instance = KappaSoftmaxStrategy()
    assert build_peer_strategy(instance) is instance
    built = build_peer_strategy({"type": "frozen_peer", "frozen": ["peer-1"], "still_scores": False})
    assert built.frozen == ("peer-1",) and built.still_scores is False
    assert build_peer_strategy({"name": "hierarchical", "params": {"update_reference": False}}).update_reference is False
    assert build_peer_strategy({"type": "frozen_peer", "path": "peer-2"}).frozen == ("peer-2",)    # the legacy alias


def test_strategy_unknown_key_rejected() -> None:
    """Behaviour change 1 (11.12): a key no constructor accepts is an error, not silently ignored."""
    with pytest.raises(ComponentParamError, match="skip_void"):
        build_peer_strategy({"type": "hierarchical", "skip_void": False})
    with pytest.raises(ComponentParamError):
        build_peer_strategy({"type": "frozen_peer", "frozenn": ["peer-1"]})
    with pytest.raises(ValidationError, match="no_such"):
        build_peer_strategy("no_such")
    with pytest.raises(ValidationError, match="peer_strategy must be"):
        build_peer_strategy(3.5)


def test_frozen_peer_takes_frozen_or_its_alias_not_both() -> None:
    with pytest.raises(ValidationError, match="not both"):
        FrozenPeerStrategy(frozen=["a"], path="b")
    assert FrozenPeerStrategy("a").frozen == ("a",)


def test_a_structural_strategy_is_returned_as_is_and_legacy_dispatched() -> None:
    class Structural:
        name = "structural"

        def on_epoch_end(self, trainer, epoch, report):
            pass

        def should_aggregate(self):
            return False

    instance = Structural()
    assert build_peer_strategy(instance) is instance
    assert not coordinates_via_context(instance)


def test_tree_view_lists_groups_in_aggregation_order() -> None:
    tree = PeerTree()
    for path in ("a", "b", "a/x", "a/y"):
        tree.add_node(path)
    view = tree_view(tree, hierarchy=True)
    assert view.labels() == ("root", "a")
    assert view.group("a") == GroupRef("a", "a", ("a/x", "a/y"), 1)
    assert view.group("root").is_root and view.paths == ("a", "b", "a/x", "a/y")
    with pytest.raises(ValidationError, match="no aggregation group 'zzz'"):
        view.group("zzz")


# -- T1: the five built-ins through the context equal the legacy dispatch ---------------------------------------
class _Legacy:
    """Wrap a built-in so the trainer legacy-dispatches it (it overrides only ``on_epoch_end``)."""

    @staticmethod
    def of(cls):
        class Wrapped(cls):
            def on_epoch_end(self, trainer, epoch, report):
                return super().on_epoch_end(trainer, epoch, report)

        Wrapped.__name__ = f"Legacy{cls.__name__}"
        return Wrapped


CASES = [
    ("independent", IndependentStrategy, {}, "flat"),
    ("cooperative", CooperativeStrategy, {}, "flat"),
    ("kappa_softmax", KappaSoftmaxStrategy, {}, "flat"),
    ("frozen_peer", FrozenPeerStrategy, {"frozen": ["peer-1"]}, "flat"),
    ("frozen_none", FrozenPeerStrategy, {}, "flat"),
    ("hierarchical", HierarchicalStrategy, {}, "hier"),
    ("hierarchical_no_skip", HierarchicalStrategy, {"skip_void_levels": False}, "hier"),
    ("kappa_on_hierarchy", KappaSoftmaxStrategy, {}, "hier"),
    ("independent_on_hierarchy", IndependentStrategy, {}, "hier"),
    ("frozen_on_hierarchy", FrozenPeerStrategy, {"frozen": ["peer-2"]}, "hier"),
]


def _build(shape, settings, tmp_path, strategy):
    if shape == "flat":
        return flat(settings, tmp_path, peer_strategy=strategy)
    return hier(settings, tmp_path, peer_strategy=strategy, roots=3)       # 3 roots: a frozen one leaves 2


@pytest.mark.parametrize("label,cls,kwargs,shape", CASES, ids=[case[0] for case in CASES])
def test_strategy_default_identity(label, cls, kwargs, shape, settings, tmp_path) -> None:
    """T1: the context bodies make the same calls, in the same order, as ``_coordinate`` did."""
    modern = run_trace(_build(shape, settings, tmp_path / "new", cls(**kwargs)))
    legacy = run_trace(_build(shape, settings, tmp_path / "old", _Legacy.of(cls)(**kwargs)))
    assert modern["calls"] == legacy["calls"], label
    assert modern["reports"] == legacy["reports"]
    assert modern["history"] == legacy["history"]
    for key, value in modern["parameters"].items():
        assert torch.equal(value, legacy["parameters"][key]), key


def test_strategy_default_identity_for_the_unconfigured_run(settings, tmp_path) -> None:
    """Nothing written: ``kappa_softmax`` on a flat tree, ``hierarchical`` on a hierarchy, no plan state."""
    trainer = flat(settings, tmp_path / "f").build()
    assert type(trainer.strategy) is KappaSoftmaxStrategy and trainer.strategy is trainer.strategy_plan.default_instance()
    assert trainer.spec.peer_strategy == "kappa_softmax"
    trainer = hier(settings, tmp_path / "h").build()
    assert type(trainer.strategy) is HierarchicalStrategy and trainer.spec.peer_strategy == "hierarchical"
    assert not trainer.strategy_plan.customized() and trainer.strategy_plan.history() == []
    assert trainer.strategy_warnings == [] and not trainer.strategy_plan.guard.covers(trainer.strategy)
    trainer.run_epoch(0)
    assert "strategy_plan" not in trainer._trainer_state() and "freeze" not in trainer._trainer_state()
    assert trainer.strategy_plan.last_audit and {e.kind for e in trainer.strategy_plan.last_audit} <= {
        "update_root", "aggregate_root", "aggregate_group", "skip"}


def test_a_default_run_makes_no_registry_build_call(settings, tmp_path, monkeypatch) -> None:
    from mactf.components.registry import ComponentRegistry

    calls = []
    original = ComponentRegistry.build
    monkeypatch.setattr(ComponentRegistry, "build",
                        lambda self, *a, **k: calls.append(self.kind) or original(self, *a, **k))
    trainer = hier(settings, tmp_path).build()
    trainer.run_epoch(0)
    assert calls == []


# -- which default? (11.5) -------------------------------------------------------------------------------------------
def test_the_default_is_authored_or_it_is_the_frameworks(settings, tmp_path) -> None:
    flat_default = {"TRAINER": {"PEER_STRATEGY": "kappa_softmax"}}
    assert default_authored(flat_default) is False                      # a plain mapping: the default value
    assert default_authored({"TRAINER": {"PEER_STRATEGY": "independent"}}) is True
    assert default_authored({"TRAINER": {"PEER_STRATEGY": {"type": "x"}}}) is True
    assert default_authored({"TRAINER": {}}) is False
    # a hierarchy whose settings merely carry the framework default keeps the hierarchical sweep (P-G2)
    trainer = hier(settings, tmp_path / "a").build()
    assert type(trainer.strategy) is HierarchicalStrategy
    # an authored value is honoured, and a code argument beats it
    trainer = hier(settings, tmp_path / "b", {"PEER_STRATEGY": "independent"}).build()
    assert type(trainer.strategy) is IndependentStrategy and trainer.spec.peer_strategy == "independent"
    trainer = hier(settings, tmp_path / "c", {"PEER_STRATEGY": "independent"}, peer_strategy="kappa_softmax").build()
    assert type(trainer.strategy) is KappaSoftmaxStrategy


def test_an_authored_kappa_softmax_on_a_hierarchy_reaches_the_root_only(settings, tmp_path) -> None:
    from p2_support import configure

    configure(settings, TRAINER={"PEER_STRATEGY": "kappa_softmax"})
    settings["_AUTHORED"]["TRAINER"] = sorted(set(settings["_AUTHORED"].get("TRAINER", [])) | {"PEER_STRATEGY"})
    trainer = hier(settings, tmp_path, epochs=1)
    trainer.build()
    assert any(note.startswith("W035") for note in trainer.strategy_warnings)       # groups nobody serves
    trainer.run_epoch(0)
    assert {tuple(report.order) for report in trainer.reports} == {("peer-1", "peer-2")}


def test_trainer_peer_strategy_setting_drives_behaviour(settings, tmp_path) -> None:
    """TRAINER.PEER_STRATEGY was validated and ignored; now it decides what the run does."""
    mixing = flat(settings, tmp_path / "m", peers=3, epochs=1)
    mixing.build()
    mixing.run_epoch(0)
    assert mixing.reports and mixing.write_count == 1
    apart = flat(settings, tmp_path / "i", {"PEER_STRATEGY": "independent"}, peers=3, epochs=1)
    apart.build()
    apart.run_epoch(0)
    assert apart.reports == [] and apart.write_count == 0                      # nobody mixed


# -- P-G6: a frozen peer below the root ------------------------------------------------------------------------------------
def test_frozen_child_gets_zero_weight(settings, tmp_path) -> None:
    """Behaviour change 2: a frozen peer below the root is excluded from its group's weights."""
    block = {"PEER_STRATEGY": "hierarchical",
             "PEER_STRATEGY_OVERRIDES": {"peer-1": {"type": "frozen_peer", "params": {"frozen": ["peer-1/sub-1"]}}}}
    trainer = hier(settings, tmp_path, block, epochs=1, fanout=3).build()      # 3 children: 2 stay eligible
    trainer.run_epoch(0)
    members = ("peer-1/sub-1", "peer-1/sub-2", "peer-1/sub-3")
    report = next(r for r in trainer.reports if tuple(r.order) == members)
    assert report.weights["peer-1/sub-1"] == 0.0
    assert sum(report.weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert "peer-1/sub-1" in report.gated_peers
    other = next(r for r in trainer.reports if tuple(r.order) == tuple(m.replace("peer-1", "peer-2") for m in members))
    assert other.weights["peer-2/sub-1"] > 0.0


def test_the_ineligible_gets_zero_in_the_context_report(settings, tmp_path) -> None:
    trainer = flat(settings, tmp_path, {"PEER_STRATEGY": {"type": "frozen_peer", "params": {"frozen": ["peer-2"]}}},
                   peers=3, epochs=1).build()
    trainer.run_epoch(0)
    assert trainer.reports[0].weights["peer-2"] == 0.0
    assert sum(trainer.reports[0].weights.values()) == pytest.approx(1.0, abs=1e-6)


# -- P-G8 -----------------------------------------------------------------------------------------------------------------------
def test_hierarchical_update_reference_false(settings, tmp_path) -> None:
    """Behaviour change 3: ``update_reference=False`` now leaves the reference EMA alone."""
    def reference_after(update):
        trainer = hier(settings, tmp_path / f"u{update}", epochs=1,
                       peer_strategy=HierarchicalStrategy(update_reference=update)).build()
        before = trainer.reference.adapter.export_parameters().detach().clone()
        trainer.run_epoch(0)
        return before, trainer.reference.adapter.export_parameters().detach().clone()

    before, moved = reference_after(True)
    assert not torch.equal(before, moved)
    before, kept = reference_after(False)
    assert torch.equal(before, kept)


# -- T7: frozen means frozen under strict --------------------------------------------------------------------------------------------
def _snapshot(trainer):
    return {path: peer.adapter.export_parameters().detach().clone() for path, peer in trainer.peers.items()}


def test_strict_freeze_a_frozen_peer_takes_no_step_and_no_write(settings, tmp_path) -> None:
    block = {"PEER_STRATEGY": {"type": "frozen_peer", "params": {"frozen": ["peer-1"]}}, "FREEZE_MODE": "strict"}
    trainer = flat(settings, tmp_path, block, peers=3, epochs=2).build()
    before = _snapshot(trainer)
    steps_before = trainer.global_step
    trainer.run_epoch(0)
    after = _snapshot(trainer)
    assert torch.equal(before["peer-1"], after["peer-1"])                   # no local step, no write-back
    assert not torch.equal(before["peer-2"], after["peer-2"]) and not torch.equal(before["peer-3"], after["peer-3"])
    assert trainer.global_step - steps_before == 2                           # two learners took their step
    assert trainer.tree["peer-1"].status == "frozen"
    assert trainer.reports[0].weights["peer-1"] == 0.0
    card = trainer._epoch_cards[-1]
    assert "peer-1" in card.peers                                            # still rolls out and is scored


def test_legacy_freeze_keeps_the_old_behaviour_and_says_so(settings, tmp_path, caplog) -> None:
    block = {"PEER_STRATEGY": {"type": "frozen_peer", "params": {"frozen": ["peer-1"]}}}
    with caplog.at_level(logging.WARNING, logger="mactf.training.trainer"):
        trainer = flat(settings, tmp_path, block, peers=3, epochs=1).build()
    assert any(note.startswith("W034") for note in trainer.strategy_warnings)
    before = _snapshot(trainer)
    trainer.run_epoch(0)
    after = _snapshot(trainer)
    assert not torch.equal(before["peer-1"], after["peer-1"])               # it still trains and receives the blend
    assert trainer.reports[0].weights["peer-1"] == 0.0                      # but is excluded from the weights


def test_strict_freeze_a_frozen_group_is_not_written(settings, tmp_path) -> None:
    block = {"PEER_STRATEGY": {"type": "frozen_peer", "params": {"frozen": ["peer-1", "peer-2", "peer-3"]}},
             "FREEZE_MODE": "strict"}
    trainer = flat(settings, tmp_path, block, peers=3, epochs=1).build()
    with pytest.raises(ImproperlyConfigured, match="E075.*nothing could learn"):
        trainer.run_epoch(0)                                                # nobody would train: refused


def test_all_frozen_is_refused(settings, tmp_path) -> None:
    freeze = FreezeSet("strict")
    tree = PeerTree.flat(["a", "b"])
    freeze.bind(tree)
    freeze.add("a", "strategy:x")
    freeze.check_learners(["a", "b"])                                       # b still learns
    freeze.add("b", "optimizer:null")
    with pytest.raises(ImproperlyConfigured, match="E075.*nothing could learn"):
        freeze.check_learners(["a", "b"])
    legacy = FreezeSet("legacy")
    legacy.bind(tree)
    legacy.add("a", "strategy:x")
    legacy.add("b", "optimizer:null")
    legacy.check_learners(["a", "b"])                                       # legacy: only retirement stops training
    legacy.add("a", "mutation:retired")
    legacy.add("b", "mutation:retired")
    with pytest.raises(ImproperlyConfigured, match="E075"):
        legacy.check_learners(["a", "b"])
    freeze.check_learners([])                                               # nothing to learn is not an error


def test_freeze_reasons_are_independent() -> None:
    freeze = FreezeSet("strict")
    tree = PeerTree.flat(["a", "b"])
    freeze.bind(tree)
    assert freeze.add("a", "strategy:one") is True
    assert freeze.add("a", "optimizer:null") is False                       # already frozen: not fresh
    assert freeze.reasons("a") == ("optimizer:null", "strategy:one") and tree["a"].status == "frozen"
    assert freeze.discard("a", "strategy:one") is False                     # another reason keeps it frozen
    assert tree["a"].status == "frozen" and "a" in freeze
    assert freeze.clear_reason("optimizer:null") == ["a"]                   # the last reason thawed it
    assert tree["a"].status == "active" and len(freeze) == 0
    assert freeze.discard("a", "strategy:one") is False                     # not held: nothing to do
    with pytest.raises(ImproperlyConfigured, match="E075.*must look like"):
        freeze.add("a", "because")
    with pytest.raises(ImproperlyConfigured, match="E075.*must look like"):
        freeze.add("a", "strategy:")
    with pytest.raises(ImproperlyConfigured, match="E075.*not a node"):
        freeze.add("ghost", "strategy:x")


def test_a_retired_peer_stays_pruned_and_legacy_effects_depend_on_the_reason() -> None:
    legacy, strict = FreezeSet("legacy"), FreezeSet("strict")
    for freeze in (legacy, strict):
        freeze.bind(PeerTree.flat(["a", "b", "c"]))
        freeze.add("a", "mutation:retired")
        freeze.add("b", "optimizer:null")
        freeze.add("c", "strategy:s")
    assert [legacy.skips_training(p) for p in "abc"] == [True, False, False]
    assert [legacy.skips_write(p) for p in "abc"] == [True, False, False]
    assert [legacy.excludes(p) for p in "abc"] == [True, False, True]       # weight 0: strategy and mutation reasons
    assert [strict.skips_training(p) for p in "abc"] == [True, True, True]
    assert legacy._tree["a"].status == "pruned" and legacy._tree["c"].status == "frozen"
    assert legacy.eligible_for(["a", "b", "c"]) == {"a": False, "b": True, "c": False}
    assert FreezeSet("legacy").eligible_for(["a"]) is None                  # nobody excluded: None, not all-True
    assert legacy.learners(["a", "b", "c"]) == ["b", "c"]
    with pytest.raises(ImproperlyConfigured, match="E070"):
        FreezeSet("sideways")


def test_freeze_state_keeps_only_what_cannot_be_rederived(caplog) -> None:
    freeze = FreezeSet("strict")
    freeze.bind(PeerTree.flat(["a", "b"]))
    freeze.add("a", "mutation:retired")
    freeze.add("b", "strategy:s")
    saved = freeze.state_dict()
    assert saved["entries"] == {"a": ["mutation:retired"], "b": ["strategy:s"]} and saved["mode"] == "strict"
    fresh = FreezeSet("legacy")
    fresh.bind(PeerTree.flat(["a"]))
    with caplog.at_level(logging.WARNING, logger="mactf.freeze"):
        notes = fresh.load_state_dict({**saved, "entries": {**saved["entries"], "gone": ["mutation:retired"]}})
    assert fresh.reasons("a") == ("mutation:retired",) and fresh.reasons("b") == ()
    assert any(note.startswith("W036") and "gone" in note for note in notes)
    assert any(note.startswith("W034") for note in notes)                    # the saved mode differs: configuration wins
    assert fresh.mode == "legacy"


def test_sync_remirrors_after_the_failed_reset() -> None:
    tree = PeerTree.flat(["a"])
    freeze = FreezeSet("strict")
    freeze.bind(tree)
    freeze.add("a", "strategy:s")
    tree.mark("a", "failed")
    freeze.sync()
    assert tree["a"].status == "failed"                                      # a failed node is not overwritten
    tree.mark("a", "active")
    freeze.sync()
    assert tree["a"].status == "frozen"


# -- P-G7: resume with a transient status -----------------------------------------------------------------------------------------------------
def test_resume_with_frozen_status(settings, tmp_path) -> None:
    """Behaviour change 4: a checkpoint written while a peer was ``frozen`` resumes."""
    block = {"PEER_STRATEGY": {"type": "frozen_peer", "params": {"frozen": ["peer-1"]}}, "FREEZE_MODE": "strict"}
    trainer = flat(settings, tmp_path / "a", block, peers=3, epochs=1).build()
    trainer.run_epoch(0)
    assert trainer.tree["peer-1"].status == "frozen"
    payload = trainer.state_dict()
    statuses = {node["path"]: node["status"] for node in payload["tree"]["nodes"]}
    assert statuses["peer-1"] == "frozen"
    fresh = flat(settings, tmp_path / "b", block, peers=3, epochs=1)
    fresh.run_id = trainer.run_id
    fresh.build()
    assert fresh.tree["peer-1"].status == "active"
    report = fresh.load_state_dict(payload)                                # used to raise ContractMismatch
    assert report.episode == trainer._epoch
    fresh.run_epoch(1)
    assert fresh.tree["peer-1"].status == "frozen"                         # re-derived by the strategy


def test_resume_with_a_failed_status_and_a_real_topology_change(settings, tmp_path) -> None:
    from mactf.orchestration.checkpoint_v2 import _topology

    base = {"nodes": [{"path": "a", "parent": None, "role": "worker", "status": "active"}]}
    for transient in ("frozen", "failed"):
        marked = {"nodes": [{**base["nodes"][0], "status": transient}]}
        assert _topology(base) == _topology(marked)
    pruned = {"nodes": [{**base["nodes"][0], "status": "pruned"}]}
    assert _topology(base) != _topology(pruned)                              # a retirement is structure
    moved = {"nodes": [{**base["nodes"][0], "parent": "z"}]}
    assert _topology(base) != _topology(moved)


# -- T12: no dead settings ----------------------------------------------------------------------------------------------------------------
def test_no_dead_strategy_settings(settings, tmp_path, component_overlay) -> None:
    block = {
        "PEER_STRATEGY": "hierarchical",
        "PEER_STRATEGY_OVERRIDES": {"peer-1": "independent"},
        "PEER_STRATEGY_OPTIONS": {"STRICT_GLOBS": False, "MIGRATE": "carry"},
        "STRATEGY_GUARD": {"MODE": "enforce", "SCOPE": "all", "PROBE": False, "CHECK_FROZEN": False},
        "FREEZE_MODE": "strict",
    }
    trainer = hier(settings, tmp_path, block).build()
    plan = trainer.strategy_plan
    assert plan.guard.describe() == {"mode": "enforce", "scope": "all", "probe": False, "check_frozen": False}
    assert plan.migrate == "carry" and plan.strict_globs is False
    assert trainer.freeze.mode == "strict"
    assert plan.override_globs() == ["peer-1"]
    assert plan.select("peer-1") == "TRAINER.PEER_STRATEGY_OVERRIDES.peer-1" and plan.select("peer-2") == DEFAULT_PATH
    described = plan.describe(["root", "peer-1"])
    assert described["guard"]["mode"] == "enforce" and described["serving"]["peer-1"].endswith("peer-1")
    assert type(trainer.strategy_plan.instance_for("peer-1")) is IndependentStrategy


@pytest.mark.parametrize("block,match", [
    ({"PEER_STRATEGY_OPTIONS": {"MIGRATE": "x"}}, "E070.*MIGRATE"),
    ({"PEER_STRATEGY_OPTIONS": {"NOPE": 1}}, "E070.*unknown keys"),
    ({"STRATEGY_GUARD": {"MODE": "x"}}, "E070.*MODE"),
    ({"STRATEGY_GUARD": {"SCOPE": "x"}}, "E070.*SCOPE"),
    ({"STRATEGY_GUARD": {"NOPE": 1}}, "E070.*unknown keys"),
    ({"FREEZE_MODE": "x"}, "E070.*FREEZE_MODE"),
    ({"PEER_STRATEGY_OVERRIDES": ["independent"]}, "mapping of glob"),
    ({"PEER_STRATEGY_OVERRIDES": {"": "independent"}}, "usable glob"),
    ({"PEER_STRATEGY_OVERRIDES": {"a": 3}}, "a name or a mapping|a spec is a name"),
])
def test_bad_strategy_settings_are_refused_at_load(settings, block, match) -> None:
    from mactf import conf

    settings["TRAINER"] = {**copy.deepcopy(dict(settings["TRAINER"])), **block}
    with pytest.raises(ImproperlyConfigured, match=match):
        conf._validate_blocks(settings)


# -- staggered mix as a real run -----------------------------------------------------------------------------------------------------------------
def test_a_registered_strategy_runs_in_a_hierarchy(settings, tmp_path, component_overlay) -> None:
    register_peer_strategy("staggered_mix", StaggeredMixStrategy, version="1", scope="run")
    block = {"PEER_STRATEGY": "hierarchical",
             "PEER_STRATEGY_OVERRIDES": {"peer-*": {"type": "staggered_mix", "params": {"period": 2}}}}
    trainer = hier(settings, tmp_path, block, epochs=4)
    trainer.build()
    mixed = []
    for epoch in range(1, 5):
        before = len(trainer.reports)
        trainer.run_epoch(epoch)
        mixed.append(sorted({tuple(r.order) for r in trainer.reports[before:]}))
    root = ("peer-1", "peer-2")
    assert all(root in groups for groups in mixed)                          # the root stays hierarchical
    children = [[g for g in groups if g != root] for groups in mixed]
    assert all(len(found) == 1 for found in children)                       # one team per epoch ...
    assert children[0] != children[1] and children[0] == children[2] and children[1] == children[3]   # ... alternating
    state = trainer.strategy_plan.state_dict()
    assert any("slots" in entry["state"] for entry in state["state"].values())
    assert trainer.failures == [] and trainer.strategy_warnings == []


def test_w032_a_group_with_every_member_frozen_is_skipped_and_logged(settings, tmp_path, caplog) -> None:
    """The built-ins skip such a group first (``ctx.skip``); ``_write`` is the backstop for a custom strategy."""
    trainer = hier(settings, tmp_path, {"FREEZE_MODE": "strict"}, epochs=1, fanout=2).build()
    members = ["peer-1/sub-1", "peer-1/sub-2"]
    for path in members:
        trainer.freeze.add(path, "strategy:test")
    before = {p: trainer.peers[p].adapter.export_parameters().detach().clone() for p in members}
    report = make_report({m: 0.5 for m in members}, members, kappa=1.0)
    with caplog.at_level(logging.WARNING, logger="mactf.training.trainer"):
        result = trainer._write(trainer.aggregator, members, report, reference=None, reference_ema=0.0)
    assert any(r.getMessage().startswith("W032") for r in caplog.records)
    assert "every member is frozen; nothing was written" in result.report.adjustments
    assert all(torch.equal(before[p], trainer.peers[p].adapter.export_parameters()) for p in members)
