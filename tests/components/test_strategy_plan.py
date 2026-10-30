"""StrategyPlan: selection, validation, live changes and resume (document 17 section 11.5, 11.9, 11.10).

Invariants T4-T6 (sweep order, one write per group), T10 (atomic swap) and T11 (restore), and the
four rules for swapping an in-flight strategy of 11.9, each with a ``unit`` test against a bare
``SlotTable`` and an ``integration`` test through the trainer.
"""

from __future__ import annotations

import copy
import logging

import pytest

from mactf.components import register_peer_strategy
from mactf.components.kinds import Boundary
from mactf.components.resolver import ComponentResolver
from mactf.components.slots import SlotTable
from mactf.components.strategy_plan import (
    DEFAULT_PATH,
    OVERRIDE_PREFIX,
    StrategyPlan,
    legacy_dispatch,
)
from mactf.coopetition.strategy import (
    GroupRef,
    HierarchicalStrategy,
    IndependentStrategy,
    KappaSoftmaxStrategy,
    PeerStrategy,
    StrategyTraits,
    TreeView,
)
from mactf.coopetition.strategy_probe import _Synthetic
from mactf.exceptions import (
    ComponentLocked,
    ComponentMissing,
    ComponentParamError,
    ImproperlyConfigured,
    StrategyGuardViolation,
)

from p3_examples import StaggeredMixStrategy
from p3_support import Recording, hier


VIEW = TreeView(
    groups=(
        GroupRef("root", None, ("team-a", "team-b"), 0),
        GroupRef("team-a", "team-a", ("team-a/x", "team-a/y"), 1),
        GroupRef("team-b", "team-b", ("team-b/x", "team-b/y"), 1),
    ),
    paths=("team-a", "team-a/x", "team-a/y", "team-b", "team-b/x", "team-b/y"),
    hierarchy=True,
)
FLAT = TreeView(groups=(GroupRef("root", None, ("a", "b"), 0),), paths=("a", "b"), hierarchy=False)


# -- strategies that record what the plan does to them -------------------------------------------------
LOG: list[tuple] = []


class Logged(PeerStrategy):
    """Mixes every group once, freezes ``freeze`` while live, and logs attach and detach."""

    name = "logged"
    traits = StrategyTraits(aggregates=True, freezes=True, walk="tree")

    def __init__(self, *, tag: str = "a", freeze: tuple = (), counter: int = 0) -> None:
        self.tag = tag
        self.freeze = tuple(freeze)
        self.counter = counter

    @property
    def reason(self) -> str:
        return "strategy:logged"

    def attach(self, host):
        if type(host) is not _Synthetic:                   # the probe's scratch instances are not live
            LOG.append(("attach", self.tag))

    def detach(self, host):
        if type(host) is not _Synthetic:
            LOG.append(("detach", self.tag))
        host.freeze.clear_reason(self.reason)

    def on_epoch_start(self, trainer, epoch):
        self.counter += 1
        for path in self.freeze:
            trainer.freeze.add(path, self.reason)

    def coordinate(self, ctx):
        for group in ctx.groups:
            if group.is_root:
                ctx.aggregate_root(ctx.update_root())
            elif not ctx.peers_void(group.members):
                ctx.aggregate_group(group)

    def validate(self, view):
        return [f"{path!r} is not a node" for path in self.freeze if path not in view.paths]

    def state_dict(self):
        return {"class": type(self).__name__, "name": self.name, "counter": self.counter, "tag": self.tag}

    def load_state_dict(self, payload):
        self.counter = int(payload.get("counter", 0))


class Other(Logged):
    name = "other_logged"


class Misbehaving(PeerStrategy):
    name = "misbehaving"
    traits = StrategyTraits(aggregates=True, walk="tree")

    def coordinate(self, ctx):
        for group in ctx.groups:
            if not group.is_root:
                ctx.aggregate_group(group)
                ctx.aggregate_group(group)                       # Y003: the probe finds it


class LegacyOnly(PeerStrategy):
    name = "legacy_only"
    traits = StrategyTraits(aggregates=False)

    def on_epoch_end(self, trainer, epoch, report):
        pass


class Raising(Logged):
    name = "raising_attach"

    def attach(self, host):
        raise RuntimeError("attach failed")


@pytest.fixture(autouse=True)
def _clean_log():
    LOG.clear()
    yield
    LOG.clear()


@pytest.fixture
def parts(component_overlay):
    for factory in (Logged, Other, Misbehaving, LegacyOnly, Raising, StaggeredMixStrategy):
        register_peer_strategy(factory.name, factory, version="1", scope="run")
    return component_overlay


def make_plan(trainer_block=None, *, view=VIEW, hierarchy=True, code_spec=None, components=None, slots=None):
    slots = slots or SlotTable(ComponentResolver())
    settings = {"TRAINER": dict(trainer_block or {}), "COMPONENTS": dict(components or {})}
    plan = StrategyPlan.from_settings(settings, slots, code_spec=code_spec, hierarchy=hierarchy)
    return plan, slots


def fingerprint(plan, slots, host):
    return (
        slots.fingerprints(),
        [r.as_dict() | {"at": 0} for r in slots.history],
        sorted(plan._adopted),
        sorted((key, id(inst)) for key, (_res, inst) in plan._group_cache.items()),
        plan.pending(),
        plan.override_globs(),
        {path: host.freeze.reasons(path) for path in VIEW.paths},
        id(plan.default_instance()),
        list(LOG),
    )


def host():
    backend = Recording(VIEW)
    return backend


# -- selection and reach ------------------------------------------------------------------------------------------
def test_the_default_for_each_tree_shape_is_the_frameworks() -> None:
    plan, slots = make_plan(hierarchy=True)
    assert type(plan.default_instance()) is HierarchicalStrategy and not plan.customized()
    plan, slots = make_plan(hierarchy=False, view=FLAT)
    assert type(plan.default_instance()) is KappaSoftmaxStrategy
    assert plan.describe(["root"])["slots"][DEFAULT_PATH]["name"] == "kappa_softmax"


def test_an_authored_value_beats_the_frameworks_default_and_the_code_beats_both(parts) -> None:
    plan, _ = make_plan({"PEER_STRATEGY": "independent"})
    assert type(plan.default_instance()) is IndependentStrategy
    instance = Logged(tag="code")
    plan, _ = make_plan({"PEER_STRATEGY": "independent"}, code_spec=instance)
    assert plan.default_instance() is instance and plan.customized()
    plan, _ = make_plan({"PEER_STRATEGY": "independent"}, code_spec="kappa_softmax")
    assert type(plan.default_instance()) is KappaSoftmaxStrategy


def test_reach_default_walks_by_trait_and_an_override_serves_exactly_its_groups(parts) -> None:
    plan, _ = make_plan({"PEER_STRATEGY": "cooperative",
                         "PEER_STRATEGY_OVERRIDES": {"team-a": {"type": "logged", "params": {"tag": "t"}}}})
    default = plan.default_instance()
    assert plan.reaches(DEFAULT_PATH, default, ROOT := VIEW.groups[0])
    assert not plan.reaches(DEFAULT_PATH, default, VIEW.groups[1])         # walk="root": a team group is not reached
    assert plan.reaches(OVERRIDE_PREFIX + "team-a", plan.instance_for("team-a"), VIEW.groups[1])
    plan2, _ = make_plan({"PEER_STRATEGY": "hierarchical"})
    assert plan2.reaches(DEFAULT_PATH, plan2.default_instance(), VIEW.groups[1])      # walk="tree"
    assert plan.select("team-a") == OVERRIDE_PREFIX + "team-a" and plan.select("team-b") == DEFAULT_PATH


def test_per_group_traits_give_one_instance_per_group(parts) -> None:
    class PerGroup(Logged):
        name = "per_group"
        traits = StrategyTraits(aggregates=True, walk="tree", per_group=True)

    register_peer_strategy("per_group", PerGroup, version="1", scope="run")
    plan, _ = make_plan({"PEER_STRATEGY": "hierarchical", "PEER_STRATEGY_OVERRIDES": {"team-*": "per_group"}})
    a, b = plan.instance_for("team-a"), plan.instance_for("team-b")
    assert a is not b and plan.instance_for("team-a") is a
    assert a in plan.live_instances() and b in plan.live_instances()
    shared, _ = make_plan({"PEER_STRATEGY": "hierarchical", "PEER_STRATEGY_OVERRIDES": {"team-*": "logged"}})
    assert shared.instance_for("team-a") is shared.instance_for("team-b")


def test_overrides_are_validated_at_build(parts) -> None:
    plan, _ = make_plan({"PEER_STRATEGY_OVERRIDES": {"nowhere": "independent"}})
    with pytest.raises(ImproperlyConfigured, match="E070.*matches no aggregation group"):
        plan.validate(VIEW, initial=True)
    assert plan.validate(VIEW, initial=False)[0].startswith("W033")
    lax, _ = make_plan({"PEER_STRATEGY_OVERRIDES": {"nowhere": "independent"},
                        "PEER_STRATEGY_OPTIONS": {"STRICT_GLOBS": False}})
    assert lax.validate(VIEW, initial=True)[0].startswith("W033")
    tie, _ = make_plan({"PEER_STRATEGY_OVERRIDES": {"te*": "independent", "*-a": "independent"}})
    with pytest.raises(ImproperlyConfigured, match="E071"):
        tie.validate(VIEW, initial=True)
    clash = TreeView(groups=(GroupRef("root", None, ("a",), 0), GroupRef("root", "root", ("root/x",), 1)),
                     paths=("root", "root/x"), hierarchy=True)
    plan, _ = make_plan()
    with pytest.raises(ImproperlyConfigured, match="E070.*collide"):
        plan.validate(clash, initial=True)


def test_e072_a_legacy_strategy_cannot_serve_one_group(parts) -> None:
    plan, _ = make_plan({"PEER_STRATEGY_OVERRIDES": {"team-a": "legacy_only"}})
    with pytest.raises(ImproperlyConfigured, match="E072.*LegacyOnly.*does not define coordinate"):
        plan.validate(VIEW, initial=True)
    class OldDefault(PeerStrategy):
        name = "old_default"

        def on_epoch_end(self, trainer, epoch, report):
            pass

    plan, _ = make_plan({"PEER_STRATEGY_OVERRIDES": {"team-a": "independent"}}, code_spec=OldDefault())
    with pytest.raises(ImproperlyConfigured, match="E072.*overrides exist"):
        plan.validate(VIEW, initial=True)
    alone, _ = make_plan(code_spec=OldDefault())
    alone.validate(VIEW, initial=True)                                      # a legacy default alone is fine


def test_e074_validate_and_probe_at_build(parts) -> None:
    plan, _ = make_plan({"PEER_STRATEGY": {"type": "logged", "params": {"freeze": ["ghost"]}}})
    with pytest.raises(ImproperlyConfigured, match="E074.*'ghost' is not a node"):
        plan.validate(VIEW, initial=True)
    plan, _ = make_plan({"PEER_STRATEGY": "misbehaving"})
    with pytest.raises(ImproperlyConfigured, match="E074.*failed its probe.*Y003"):
        plan.validate(VIEW, initial=True)
    off, _ = make_plan({"PEER_STRATEGY": "misbehaving", "STRATEGY_GUARD": {"PROBE": False}})
    off.validate(VIEW, initial=True)                                        # the probe is a switch, the guard still runs
    class Nothing(PeerStrategy):
        name = "nothing"

    plan, _ = make_plan(code_spec=Nothing())
    with pytest.raises(ImproperlyConfigured, match="E074.*defines neither"):
        plan.validate(VIEW, initial=True)
    with pytest.raises(ComponentMissing):
        make_plan({"PEER_STRATEGY": "no_such"})
    with pytest.raises(ComponentParamError):
        make_plan({"PEER_STRATEGY": {"type": "hierarchical", "params": {"skip_void": 1}}})


def test_the_default_plan_probes_nothing_and_builds_no_scratch_instances(monkeypatch) -> None:
    from mactf.components.registry import ComponentRegistry

    calls = []
    original = ComponentRegistry.build
    monkeypatch.setattr(ComponentRegistry, "build", lambda self, *a, **k: calls.append(self.kind) or original(self, *a, **k))
    plan, _ = make_plan()
    plan.validate(VIEW, initial=True)
    assert calls == []


# -- the sweep (T4, T5, T6) --------------------------------------------------------------------------------------------
def test_run_sweeps_deepest_group_first_and_precomputes_the_root_for_hierarchical() -> None:
    plan, _ = make_plan()
    backend = Recording(VIEW)
    backend._epoch_report = None
    root = plan.run(backend, 1, VIEW)
    events = [(e.kind, e.group) for e in plan.last_audit]
    assert events == [("update_root", "root"), ("aggregate_group", "team-b"), ("aggregate_group", "team-a"),
                      ("aggregate_root", "root")]
    assert backend.writes == [("team-b", ("team-b/x", "team-b/y")), ("team-a", ("team-a/x", "team-a/y")), ("root", True)]


def test_run_in_a_legacy_default_calls_the_legacy_dispatch(parts) -> None:
    calls = []

    class Legacy(PeerStrategy):
        name = "legacy_default"
        traits = StrategyTraits(aggregates=True)

        def on_epoch_end(self, trainer, epoch, report):
            calls.append(("end", report is not None))

    class Host(Recording):
        def update_weights(self, eligible=None):
            calls.append(("update", eligible))
            return super().update_weights(eligible)

        def aggregate(self, report=None, *, update_reference=True):
            calls.append(("aggregate", report is not None))
            return super().aggregate(report, update_reference=update_reference)

    plan, _ = make_plan(code_spec=Legacy(), hierarchy=False, view=FLAT)
    backend = Host(FLAT)
    plan.run(backend, 1, FLAT)
    assert calls == [("update", None), ("aggregate", True), ("end", True)]     # a should_aggregate() legacy strategy


def test_legacy_dispatch_covers_every_branch() -> None:
    calls = []

    class Trainer:
        def update_weights(self, eligible=None):
            calls.append("update")
            return "report"

        def aggregate(self, report=None):
            calls.append("aggregate")

    class Agg(PeerStrategy):
        name = "agg"
        traits = StrategyTraits(aggregates=True)

        def on_epoch_end(self, trainer, epoch, report):
            calls.append(("end", report))

    class NoAgg(Agg):
        traits = StrategyTraits(aggregates=False)

    legacy_dispatch(Trainer(), 1, Agg())
    legacy_dispatch(Trainer(), 1, NoAgg())
    assert calls == ["update", "aggregate", ("end", "report"), ("end", None)]


def test_epoch_start_reaches_every_live_instance_and_per_group_ones_exist_first(parts) -> None:
    class PerGroup(Logged):
        name = "per_group"
        traits = StrategyTraits(aggregates=True, walk="tree", per_group=True)

    register_peer_strategy("per_group", PerGroup, version="1", scope="run")
    plan, _ = make_plan({"PEER_STRATEGY": "hierarchical", "PEER_STRATEGY_OVERRIDES": {"team-*": "per_group"}})
    backend = Recording(VIEW)
    plan.epoch_start(backend, 1, VIEW)
    counters = sorted(inst.counter for inst in plan.live_instances() if isinstance(inst, PerGroup))
    assert counters == [1, 1, 1]                      # the slot's own instance and one per team group


# -- swapping an in-flight strategy: the four rules (11.9) --------------------------------------------------------------
# Rule 1: the change takes effect at the next episode boundary.
def test_rule_1_unit_a_request_changes_nothing_until_the_boundary_and_step_is_refused(parts) -> None:
    plan, slots = make_plan({"PEER_STRATEGY": "logged"})
    backend = host()
    old = plan.default_instance()
    plan.request(None, "other_logged")
    assert plan.default_instance() is old and plan.pending() == {DEFAULT_PATH: "set"}
    with pytest.raises(ComponentLocked, match="episode boundaries"):
        plan.set(None, "other_logged", view=VIEW, host=backend, boundary=Boundary.STEP)
    assert plan.default_instance() is old
    (glob, value, mode), = plan.take_pending()
    plan.set(glob, value, view=VIEW, host=backend, mode=mode)                 # the boundary applies it
    assert type(plan.default_instance()) is Other


def test_rule_1_integration_a_swap_requested_mid_epoch_lands_next_epoch(settings, tmp_path, parts) -> None:
    seen = []

    class SelfSwapping(Logged):
        name = "self_swapping"

        def coordinate(self, ctx):
            if ctx.epoch == 1 and ctx.group.is_root and getattr(self, "host", None) is not None:
                self.host.request_peer_strategy(None, "independent")
            if getattr(self, "host", None) is not None:        # not the probe's scratch run
                seen.append((ctx.epoch, ctx.group.label))
            super().coordinate(ctx)

        def attach(self, host):
            self.host = host

    register_peer_strategy("self_swapping", SelfSwapping, version="1", scope="run")
    trainer = hier(settings, tmp_path, {"PEER_STRATEGY": "self_swapping"}, epochs=3).build()
    trainer.run_epoch(1)
    assert isinstance(trainer.strategy, SelfSwapping)                         # this epoch finished under the old one
    assert [label for epoch, label in seen] == ["peer-2", "peer-1", "root"] and trainer.write_count == 3
    trainer.run_epoch(2)
    assert isinstance(trainer.strategy, IndependentStrategy)                  # the next epoch runs the new one
    assert trainer.write_count == 3                                           # and the independent strategy writes nothing
    assert trainer.strategy_plan.history()[-1].to_name == "independent"


# Rule 2: detach before attach; a peer frozen twice stays frozen.
def test_rule_2_unit_detach_runs_before_attach_and_a_double_frozen_peer_stays_frozen(parts) -> None:
    plan, slots = make_plan({"PEER_STRATEGY": {"type": "logged", "params": {"tag": "old", "freeze": ["team-a/x"]}}})
    backend = host()
    plan.default_instance().attach(backend)
    plan.epoch_start(backend, 1, VIEW)                                        # the old strategy freezes team-a/x
    backend.freeze.add("team-a/x", "optimizer:null")                          # ... and the optimizer plan does too
    LOG.clear()
    plan.set(None, {"type": "other_logged", "params": {"tag": "new"}}, view=VIEW, host=backend)
    assert LOG == [("detach", "old"), ("attach", "new")]
    assert backend.freeze.reasons("team-a/x") == ("optimizer:null",)          # its own reason is gone, the other stays
    assert backend.tree["team-a/x"].status == "frozen"


def test_rule_2_unit_a_raising_detach_does_not_undo_the_swap(parts, caplog) -> None:
    class BadDetach(Logged):
        name = "bad_detach"

        def detach(self, host):
            raise RuntimeError("cannot release")

    register_peer_strategy("bad_detach", BadDetach, version="1", scope="run")
    plan, _ = make_plan({"PEER_STRATEGY": "bad_detach"})
    with caplog.at_level(logging.ERROR, logger="mactf.strategy"):
        plan.set(None, "logged", view=VIEW, host=host())
    assert type(plan.default_instance()) is Logged
    assert any("detach of BadDetach raised" in record.message for record in caplog.records)


def test_rule_2_unit_a_raising_attach_is_reported_not_raised(parts) -> None:
    plan, _ = make_plan({"PEER_STRATEGY": "logged"})
    plan.set(None, "raising_attach", view=VIEW, host=host())
    assert type(plan.default_instance()).__name__ == "Raising"                # the swap is done (it cannot be undone)
    errors = plan.drain_errors()
    assert len(errors) == 1 and "attach of Raising raised RuntimeError" in errors[0] and plan.drain_errors() == []


def test_rule_2_integration_a_strict_frozen_peer_is_thawed_by_the_swap_out(settings, tmp_path) -> None:
    block = {"PEER_STRATEGY": {"type": "frozen_peer", "params": {"frozen": ["peer-1"]}}, "FREEZE_MODE": "strict"}
    from p3_support import flat

    trainer = flat(settings, tmp_path, block, peers=3, epochs=3).build()
    trainer.run_epoch(0)
    assert trainer.freeze.reasons("peer-1") == ("strategy:frozen_peer",) and trainer.tree["peer-1"].status == "frozen"
    trainer.freeze.add("peer-1", "optimizer:null")                            # a second reason
    trainer.swap_peer_strategy(None, "kappa_softmax")
    assert trainer.freeze.reasons("peer-1") == ("optimizer:null",)            # detach released only its own
    assert trainer.tree["peer-1"].status == "frozen"
    trainer.freeze.discard("peer-1", "optimizer:null")
    assert trainer.tree["peer-1"].status == "active"


# Rule 3: state is carried only for the same class under MIGRATE=carry; fail refuses.
def test_rule_3_unit_carry_moves_state_for_the_same_class_only(parts, caplog) -> None:
    def armed(mode):
        plan, slots = make_plan({"PEER_STRATEGY": {"type": "logged", "params": {"counter": 5}},
                                 "PEER_STRATEGY_OPTIONS": {"MIGRATE": mode}})
        return plan, host()

    plan, backend = armed("carry")
    plan.set(None, {"type": "logged", "params": {"tag": "new"}}, view=VIEW, host=backend)
    assert plan.default_instance().counter == 5 and plan.default_instance().tag == "new"
    plan, backend = armed("carry")
    with caplog.at_level(logging.WARNING, logger="mactf.strategy"):
        plan.set(None, "other_logged", view=VIEW, host=backend)               # another class: nothing to carry
    assert plan.default_instance().counter == 0
    assert any("W031" in record.message and "carry" in record.message for record in caplog.records)
    plan, backend = armed("reset")
    plan.set(None, {"type": "logged", "params": {"tag": "new"}}, view=VIEW, host=backend)
    assert plan.default_instance().counter == 0
    plan, backend = armed("reset")
    plan.set(None, {"type": "logged", "params": {"tag": "new"}}, view=VIEW, host=backend, mode="carry")   # per call
    assert plan.default_instance().counter == 5


def test_rule_3_unit_fail_refuses_a_swap_that_would_drop_state(parts) -> None:
    plan, slots = make_plan({"PEER_STRATEGY": {"type": "logged", "params": {"counter": 5}},
                             "PEER_STRATEGY_OPTIONS": {"MIGRATE": "fail"}})
    backend = host()
    before = fingerprint(plan, slots, backend)
    with pytest.raises(ImproperlyConfigured, match="E074.*discard its state.*MIGRATE is 'fail'"):
        plan.set(None, "other_logged", view=VIEW, host=backend)
    assert fingerprint(plan, slots, backend) == before
    with pytest.raises(ImproperlyConfigured, match="E074.*migrate must be one of"):
        plan.set(None, "other_logged", view=VIEW, host=backend, mode="merge")
    plain, _ = make_plan({"PEER_STRATEGY": "independent", "PEER_STRATEGY_OPTIONS": {"MIGRATE": "fail"}})
    plain.set(None, "kappa_softmax", view=VIEW, host=backend)                  # no state to drop: allowed
    with pytest.raises(ImproperlyConfigured, match="E070.*MIGRATE"):
        make_plan({"PEER_STRATEGY_OPTIONS": {"MIGRATE": "x"}})


def test_rule_3_integration_carry_through_the_trainer(settings, tmp_path, parts) -> None:
    trainer = hier(settings, tmp_path, {"PEER_STRATEGY": "staggered_mix"}, epochs=3).build()
    trainer.run_epoch(1)
    slots_before = dict(trainer.strategy._slots)
    assert slots_before
    trainer.request_peer_strategy(None, {"type": "staggered_mix", "params": {"period": 5}}, mode="carry")
    trainer.run_epoch(2)
    assert isinstance(trainer.strategy, StaggeredMixStrategy) and trainer.strategy.period == 5
    assert all(trainer.strategy._slots[label] == slot for label, slot in slots_before.items())
    trainer.swap_peer_strategy(None, {"type": "staggered_mix", "params": {"period": 2}}, mode="reset")
    assert trainer.strategy._slots == {}


# Rule 4: a failing swap changes nothing; a request fails early for what is knowable without the tree.
@pytest.mark.parametrize("step,candidate,expected", [
    ("unknown", "no_such", "no_such"),
    ("params", {"type": "staggered_mix", "params": {"nope": 1}}, "nope"),
    ("build", {"type": "staggered_mix", "params": {"period": 0}}, "period"),
    ("validate", {"type": "logged", "params": {"freeze": ["ghost"]}}, "E074.*'ghost' is not a node"),
    ("probe", "misbehaving", "E074.*failed its probe.*Y003"),
    ("placement", "legacy_only", "E072"),
    ("migrate", "other_logged", "E074.*MIGRATE is 'fail'"),
    ("lock", "other_logged", "locked"),
    ("commit", "other_logged", "injected commit fault"),
])
def test_strategy_swap_atomic(parts, monkeypatch, step, candidate, expected) -> None:
    """T10 / rule 4: a swap that fails at any step leaves the old strategy, its state and the freeze set."""
    plan, slots = make_plan({"PEER_STRATEGY": {"type": "logged", "params": {"counter": 3, "freeze": ["team-a/x"]}},
                             "PEER_STRATEGY_OVERRIDES": {"team-b": "independent"}})
    backend = host()
    plan.epoch_start(backend, 1, VIEW)                                        # the strategy freezes team-a/x
    LOG.clear()
    kwargs = {}
    glob = None
    if step == "migrate":
        kwargs["mode"] = "fail"
    if step == "lock":
        monkeypatch.setattr(SlotTable, "_slot_locked", staticmethod(lambda path: True))
    if step == "commit":
        def boom(self, *args, **kw):
            raise RuntimeError("injected commit fault")
        monkeypatch.setattr(SlotTable, "swap", boom)
    if step == "placement":
        glob = "team-a"                                                       # a legacy strategy as an override
    before = fingerprint(plan, slots, backend)
    state_before = plan.state_dict()
    with pytest.raises(Exception, match=expected):
        plan.set(glob, candidate, view=VIEW, host=backend, **kwargs)
    assert fingerprint(plan, slots, backend) == before                       # incl. freeze reasons, attach/detach log
    assert plan.state_dict() == state_before
    assert LOG == []                                                          # neither detach nor attach ran


def test_strategy_swap_atomic_through_the_trainer(settings, tmp_path, parts) -> None:
    block = {"PEER_STRATEGY": {"type": "logged", "params": {"freeze": ["peer-1/sub-1"]}}, "FREEZE_MODE": "strict"}
    trainer = hier(settings, tmp_path, block, epochs=3).build()
    trainer.run_epoch(1)
    before_reasons = {p: trainer.freeze.reasons(p) for p in trainer.tree._nodes}
    old = trainer.strategy
    LOG.clear()
    with pytest.raises(ImproperlyConfigured, match="E074.*probe"):
        trainer.swap_peer_strategy(None, "misbehaving")
    assert trainer.strategy is old and trainer.strategy_plan.default_instance() is old
    assert {p: trainer.freeze.reasons(p) for p in trainer.tree._nodes} == before_reasons
    assert LOG == [] and trainer.strategy_plan.history() == []
    trainer.run_epoch(2)                                                      # the run carries on unharmed
    assert trainer.failures == []


def test_a_request_fails_at_request_time_for_what_needs_no_tree(parts) -> None:
    plan, _ = make_plan({"PEER_STRATEGY": "logged"})
    for bad in ("no_such", {"type": "staggered_mix", "params": {"zzz": 1}}):
        with pytest.raises(Exception):
            plan.request(None, bad)
    assert plan.pending() == {}
    with pytest.raises(ImproperlyConfigured, match="E074.*migrate"):
        plan.request(None, "other_logged", mode="merge")
    with pytest.raises(ComponentMissing):
        plan.request_remove("nothing")
    plan.request(None, "other_logged")                                        # the tree-dependent checks wait
    assert plan.pending() == {DEFAULT_PATH: "set"}


def test_a_failed_queued_change_is_recorded_and_the_run_goes_on(settings, tmp_path, parts) -> None:
    trainer = hier(settings, tmp_path, {"PEER_STRATEGY": "logged"}, epochs=3).build()
    trainer.request_peer_strategy(None, {"type": "logged", "params": {"freeze": ["ghost"]}})   # fails at the boundary
    trainer.run_epoch(1)
    assert any("queued peer_strategy change" in f and "E074" in f for f in trainer.failures)
    assert type(trainer.strategy) is Logged and trainer.strategy.tag == "a"
    trainer.request_peer_strategy("peer-1", "independent")
    trainer.run_epoch(2)
    assert trainer.strategy_plan.override_globs() == ["peer-1"]


def test_adding_and_removing_an_override_through_the_trainer(settings, tmp_path, parts) -> None:
    trainer = hier(settings, tmp_path, {"PEER_STRATEGY": "hierarchical"}, epochs=3).build()
    assert trainer.swap_peer_strategy("peer-1", "independent") is None         # an addition records no swap
    trainer.run_epoch(1)
    assert {tuple(r.order) for r in trainer.reports} == {("peer-1", "peer-2"), ("peer-2/sub-1", "peer-2/sub-2")}
    record = trainer.remove_peer_override("peer-1")
    assert record.to_name == "<removed>" and trainer.strategy_plan.override_globs() == []
    with pytest.raises(ComponentMissing):
        trainer.remove_peer_override("peer-1")
    before = len(trainer.reports)
    trainer.run_epoch(2)
    assert ("peer-1/sub-1", "peer-1/sub-2") in {tuple(r.order) for r in trainer.reports[before:]}


def test_remove_that_creates_a_tie_is_refused(parts) -> None:
    plan, slots = make_plan({"PEER_STRATEGY_OVERRIDES": {"team-a": "independent", "te*": "independent", "*-a": "independent"}})
    backend = host()
    before = fingerprint(plan, slots, backend)
    with pytest.raises(ImproperlyConfigured, match="E071"):
        plan.remove("team-a", view=VIEW, host=backend)
    assert fingerprint(plan, slots, backend) == before


# -- resume (T11) ------------------------------------------------------------------------------------------------------
def test_resume_restores_slots_state_and_per_group_state_lazily(parts) -> None:
    class PerGroup(Logged):
        name = "per_group"
        traits = StrategyTraits(aggregates=True, walk="tree", per_group=True)

    register_peer_strategy("per_group", PerGroup, version="1", scope="run")
    block = {"PEER_STRATEGY": {"type": "logged", "params": {"counter": 4}},
             "PEER_STRATEGY_OVERRIDES": {"team-*": "per_group"}}
    plan, _ = make_plan(block)
    backend = host()
    plan.epoch_start(backend, 1, VIEW)
    plan.instance_for("team-a").counter = 9
    saved = plan.state_dict()
    assert saved["state"][f"{DEFAULT_PATH}|*"]["state"]["counter"] == 5
    assert saved["state"][f"{OVERRIDE_PREFIX}team-*|team-a"]["state"]["counter"] == 9

    fresh, _ = make_plan(block)
    assert fresh.load_state_dict(saved) == []
    assert fresh.default_instance().counter == 5                               # per-slot state: now
    assert fresh._group_cache == {}                                            # per-group state: on first use
    assert fresh.instance_for("team-a").counter == 9
    assert fresh.instance_for("team-b").counter == 1                           # b had its own, fresh state


def test_resume_a_changed_default_is_w031_and_the_saved_plan_wins(parts) -> None:
    plan, _ = make_plan({"PEER_STRATEGY": {"type": "logged", "params": {"tag": "saved"}}})
    saved = plan.state_dict()
    other, _ = make_plan({"PEER_STRATEGY": "other_logged"})
    notes = other.load_state_dict(saved)
    assert any(n.startswith("W031") and DEFAULT_PATH in n for n in notes)
    assert type(other.default_instance()) is Logged and other.default_instance().tag == "saved"
    strict, _ = make_plan({"PEER_STRATEGY": "other_logged"})
    with pytest.raises(ImproperlyConfigured, match="E076.*STRICT_RESTORE"):
        strict.load_state_dict(saved, strict=True)


def test_resume_overrides_a_removed_one_returns_and_a_new_one_goes(parts) -> None:
    plan, _ = make_plan({"PEER_STRATEGY": "logged", "PEER_STRATEGY_OVERRIDES": {"team-a": "independent"}})
    saved = plan.state_dict()
    fresh, _ = make_plan({"PEER_STRATEGY": "logged", "PEER_STRATEGY_OVERRIDES": {"team-b": "independent"}})
    notes = fresh.load_state_dict(saved)
    assert fresh.override_globs() == ["team-a"]                                # the saved plan won both ways
    assert sum(n.startswith("W031") for n in notes) == 2
    strict, _ = make_plan({"PEER_STRATEGY": "logged", "PEER_STRATEGY_OVERRIDES": {"team-b": "independent"}})
    with pytest.raises(ImproperlyConfigured, match="E076"):
        strict.load_state_dict(saved, strict=True)


def test_resume_state_for_another_class_is_dropped(parts, caplog) -> None:
    plan, _ = make_plan({"PEER_STRATEGY": "logged"})
    plan.default_instance().counter = 4
    saved = plan.state_dict()
    assert saved["state"][f"{DEFAULT_PATH}|*"]["state"]["counter"] == 4
    saved["state"][f"{DEFAULT_PATH}|*"]["class"] = "SomethingElse"
    fresh, _ = make_plan({"PEER_STRATEGY": "logged"})
    with caplog.at_level(logging.WARNING, logger="mactf.strategy"):
        notes = fresh.load_state_dict(saved)
    assert any("does not fit the live strategy" in n for n in notes) and fresh.default_instance().counter == 0


def test_resume_an_in_code_instance_is_never_rebuilt(parts) -> None:
    plan, _ = make_plan(code_spec=Logged(tag="code"))
    saved = plan.state_dict()
    assert "adopted" in saved["slots"][DEFAULT_PATH]
    fresh, _ = make_plan({"PEER_STRATEGY": "independent"})
    notes = fresh.load_state_dict(saved)
    assert any("in-code instance" in n for n in notes) and type(fresh.default_instance()) is IndependentStrategy


def test_resume_through_the_trainer_keeps_state_and_a_swapped_strategy(settings, tmp_path, parts) -> None:
    block = {"PEER_STRATEGY": "hierarchical", "PEER_STRATEGY_OVERRIDES": {"peer-*": "staggered_mix"}}
    trainer = hier(settings, tmp_path / "a", block, epochs=3).build()
    trainer.run_epoch(1)
    trainer.swap_peer_strategy("peer-*", {"type": "staggered_mix", "params": {"period": 7}})
    trainer.run_epoch(2)
    state = trainer._trainer_state()
    assert state["strategy_plan"]["history"] and "staggered" in str(state["strategy_plan"]["slots"])
    payload = trainer.state_dict()
    fresh = hier(settings, tmp_path / "b", block, epochs=3)
    fresh.run_id = trainer.run_id
    fresh.build()
    fresh.load_state_dict(payload)
    live = fresh.strategy_plan.instance_for("peer-1")
    assert isinstance(live, StaggeredMixStrategy) and live.period == 7                 # the swapped strategy won
    assert live._slots == trainer.strategy_plan.instance_for("peer-1")._slots
    assert any(n.startswith("W031") for n in fresh.strategy_warnings)


def test_a_retired_peer_is_persisted_and_a_strategy_freeze_is_rederived(settings, tmp_path, parts) -> None:
    block = {"PEER_STRATEGY": {"type": "logged", "params": {"freeze": ["peer-1/sub-1"]}}}
    trainer = hier(settings, tmp_path / "a", block, epochs=3, fanout=3).build()
    trainer.run_epoch(1)
    assert trainer.freeze.reasons("peer-1/sub-1") == ("strategy:logged",)
    trainer.freeze.add("peer-2/sub-2", "mutation:retired")
    state = trainer._trainer_state()
    assert state["freeze"]["entries"]["peer-2/sub-2"] == ["mutation:retired"]
    fresh = hier(settings, tmp_path / "b", block, epochs=3, fanout=3).build()
    fresh.freeze.load_state_dict(state["freeze"])
    assert fresh.freeze.reasons("peer-2/sub-2") == ("mutation:retired",)
    assert fresh.freeze.reasons("peer-1/sub-1") == ()                       # re-added by on_epoch_start, not stored
    assert fresh.tree["peer-2/sub-2"].status == "pruned"
    fresh.run_epoch(1)
    assert fresh.freeze.reasons("peer-1/sub-1") == ("strategy:logged",)


# -- the carried state is proven before it is written ----------------------------------------------------------------
def test_a_candidate_that_cannot_take_the_carried_state_changes_nothing(parts) -> None:
    class Picky(Logged):
        name = "picky_state"

        def load_state_dict(self, payload):
            raise ValueError("wrong shape")

    register_peer_strategy("picky_state", Picky, version="1", scope="run")
    plan, slots = make_plan({"PEER_STRATEGY": "picky_state", "STRATEGY_GUARD": {"PROBE": False}})
    plan.default_instance().counter = 7
    backend = host()
    before = fingerprint(plan, slots, backend)
    with pytest.raises(ImproperlyConfigured, match="E074.*cannot take the carried state.*wrong shape.*nothing was changed"):
        plan.set(None, "picky_state", view=VIEW, host=backend, mode="carry")
    assert fingerprint(plan, slots, backend) == before


def test_a_load_that_fails_inside_the_swap_leaves_the_old_strategy_running(parts) -> None:
    """The ``migrate`` step runs before the slot is rewritten (``SlotTable.swap``); a failure keeps the old one."""
    from mactf.exceptions import ComponentMigrationError

    class Flaky(Logged):
        name = "flaky_state"
        loads = 0

        def load_state_dict(self, payload):
            type(self).loads += 1
            if type(self).loads > 1:                       # the pre-check passes, the live load does not
                raise RuntimeError("second load fails")

    register_peer_strategy("flaky_state", Flaky, version="1", scope="run")
    plan, slots = make_plan({"PEER_STRATEGY": "flaky_state", "STRATEGY_GUARD": {"PROBE": False}})
    plan.default_instance().counter = 3
    backend = host()
    before = fingerprint(plan, slots, backend)
    with pytest.raises(ComponentMigrationError, match="the old component keeps running"):
        plan.set(None, "flaky_state", view=VIEW, host=backend, mode="carry")
    assert fingerprint(plan, slots, backend) == before
    assert plan.default_instance().counter == 3


def test_a_stale_group_state_that_cannot_load_is_dropped_not_fatal(parts, caplog) -> None:
    class Strict(Logged):
        name = "strict_load"

        def load_state_dict(self, payload):
            raise KeyError("shape changed")

    register_peer_strategy("strict_load", Strict, version="1", scope="run")

    class PerGroup(Strict):
        name = "strict_per_group"
        traits = StrategyTraits(aggregates=True, walk="tree", per_group=True)

    register_peer_strategy("strict_per_group", PerGroup, version="1", scope="run")
    block = {"PEER_STRATEGY": "hierarchical", "PEER_STRATEGY_OVERRIDES": {"team-*": "strict_per_group"}}
    plan, _ = make_plan(block)
    saved = plan.state_dict()
    saved["state"] = {f"{OVERRIDE_PREFIX}team-*|team-a": {"class": "PerGroup", "state": {"counter": 1}}}
    fresh, _ = make_plan(block)
    fresh.load_state_dict(saved)
    with caplog.at_level(logging.WARNING, logger="mactf.strategy"):
        instance = fresh.instance_for("team-a")                    # does not raise
    assert instance is not None and any("could not be loaded" in r.message for r in caplog.records)
