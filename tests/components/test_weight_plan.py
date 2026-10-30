"""WeightPlan, KappaPlan and the weight update wired into the trainer (document 17 section 10).

Invariants A1-A9 of 10.2: default identity, legal weights, the framework owns the bounds, one
instance per (slot, group), deterministic selection, atomic swap, a rejected proposal costs
nothing, restore is exact or loud, fail at build.
"""

from __future__ import annotations

import copy
import logging

import pytest
import torch

from mactf.components import register_aggregator, register_weight_update
from mactf.components.kappa_plan import KAPPA_KEYS, KappaPlan
from mactf.components.kinds import Boundary
from mactf.components.registry import ComponentRegistry
from mactf.components.resolver import ComponentResolver, ComponentSpec
from mactf.components.slots import SlotTable
from mactf.components.weight_plan import (
    DEFAULT_PATH,
    OVERRIDE_PREFIX,
    WeightPlan,
    group_label,
    pick_glob,
)
from mactf.coopetition.aggregation import Aggregator, build_aggregator, verify_aggregator
from mactf.coopetition.temperature import AdaptiveKappa
from mactf.coopetition.weights import (
    GroupView,
    KappaSoftmaxAverage,
    UniformAverage,
    WeightStrategy,
    WeightTraits,
    build_weight_update,
    make_report,
)
from mactf.exceptions import (
    ComponentLocked,
    ComponentMissing,
    ComponentParamError,
    ComponentSpecError,
    ImproperlyConfigured,
    ValidationError,
    WeightGuardViolation,
)

from p2_support import configure, hierarchy_trainer, make_trainer, tune
from p3_examples import EmaSoftmax


# -- fixtures ----------------------------------------------------------------------------
class SumsToThree(WeightStrategy):
    name = "sums_to_three"

    def apply(self, vectors, scores, *, kappa, level=0, eligible=None, confidence=None):
        names = sorted(vectors)
        return make_report({n: 3.0 / len(names) for n in names}, names, kappa=kappa, eligible=eligible)


class Raises(WeightStrategy):
    name = "raises"

    def apply(self, vectors, scores, *, kappa, level=0, eligible=None, confidence=None):
        raise KeyError("third-party bug")


class Picky(WeightStrategy):
    """Refuses groups of fewer than ``minimum`` peers (``validate``)."""

    name = "picky"

    def __init__(self, *, minimum: int = 3) -> None:
        self.minimum = minimum

    def validate(self, group: GroupView):
        return [f"needs {self.minimum} peers"] if len(group.members) < self.minimum else []

    def apply(self, vectors, scores, *, kappa, level=0, eligible=None, confidence=None):
        names = sorted(vectors)
        return make_report({n: 1.0 / len(names) for n in names}, names, kappa=kappa, eligible=eligible)


class Advancing(WeightStrategy):
    """Advances the controller (as a strategy composing ``compute_weights`` would) then misbehaves."""

    name = "advancing"

    def __init__(self, aggregator: Aggregator | None = None) -> None:
        self.aggregator = aggregator

    def apply(self, vectors, scores, *, kappa, level=0, eligible=None, confidence=None):
        names = sorted(vectors)
        self.aggregator.compute_weights(names, [float(scores.get(n, 0.0)) for n in names])    # advance=True
        return make_report({n: 5.0 for n in names}, names, kappa=kappa, eligible=eligible)   # then garbage


@pytest.fixture
def parts(component_overlay):
    for factory in (EmaSoftmax, SumsToThree, Raises, Picky, Advancing):
        register_weight_update(factory.name, factory, version="1", scope="run")
    return component_overlay


def _aggregator(peers: int = 2) -> Aggregator:
    return Aggregator(controller=AdaptiveKappa(num_peers=peers))


def _views(*groups):
    """``_views(("root", None, ("a", "b")), ("team-a", "team-a", ("team-a/x", "team-a/y")))``."""
    return [GroupView(label=label, parent=parent, members=tuple(members),
                      level=0 if parent is None else 1, floor=0.02, cap=0.6)
            for label, parent, members in groups]


VIEWS = _views(("root", None, ("team-a", "team-b")),
               ("team-a", "team-a", ("team-a/x", "team-a/y", "team-a/z")),
               ("team-b", "team-b", ("team-b/x", "team-b/y", "team-b/z")))


def _plan(trainer_block=None, *, aggregator=None, slots=None, code_spec=None, components=None):
    slots = slots or SlotTable(ComponentResolver())
    settings = {"TRAINER": dict(trainer_block or {}), "COMPONENTS": dict(components or {})}
    plan = WeightPlan.from_settings(
        settings, slots, aggregator=aggregator or _aggregator(), code_spec=code_spec
    )
    return plan, slots


def _fingerprint(plan: WeightPlan, slots: SlotTable):
    """Everything an atomic swap must leave alone."""
    return (
        slots.fingerprints(),
        [record.as_dict() | {"at": 0} for record in slots.history],
        sorted(plan._adopted),
        {label: id(built.instance) for label, built in plan._cache.items()},
        copy.deepcopy(plan._carry),
        plan.pending(),
        plan.override_globs(),
    )


# -- selection (A5) -----------------------------------------------------------------------------
def test_override_specificity_exact_label_wins_over_any_glob() -> None:
    assert pick_glob("team-a", ["team-a", "team-*", "*"], what="x", code="E061") == "team-a"


def test_override_specificity_the_longest_literal_prefix_wins() -> None:
    assert pick_glob("team-a", ["team-*", "t*", "*"], what="x", code="E061") == "team-*"
    assert pick_glob("zzz", ["team-*"], what="x", code="E061") is None


def test_override_specificity_a_tie_is_e061() -> None:
    with pytest.raises(ImproperlyConfigured, match="E061.*equally specific"):
        pick_glob("abab", ["ab*", "*ab"], what="weight_update overrides", code="E061")


def test_the_plan_selects_by_label(parts) -> None:
    plan, _ = _plan({"WEIGHT_UPDATE_OVERRIDES": {"team-a": "uniform", "team-*": "score_weighted"}})
    assert plan.select("team-a") == OVERRIDE_PREFIX + "team-a"
    assert plan.select("team-b") == OVERRIDE_PREFIX + "team-*"
    assert plan.select("root") == DEFAULT_PATH
    assert group_label(None) == "root" and group_label("team-a") == "team-a"


# -- instances (A4) -----------------------------------------------------------------------------------
def test_group_state_isolated(parts) -> None:
    plan, _ = _plan({"WEIGHT_UPDATE": {"type": "ema_softmax", "params": {"alpha": 0.5}}})
    aggregator = _aggregator()
    a, b = plan.strategy_for("team-a", aggregator), plan.strategy_for("team-b", aggregator)
    assert a is not b and isinstance(a, EmaSoftmax) and a.alpha == 0.5
    vectors = {name: torch.zeros(2) for name in ("x", "y")}
    a.apply(vectors, {"x": 1.0, "y": 0.0}, kappa=1.0)
    assert a._ema and not b._ema                         # state never crosses groups
    assert plan.strategy_for("team-a", aggregator) is a                  # cached
    assert plan.strategy_for("team-a", _aggregator()) is not a           # a new aggregator: a new instance


def test_the_default_is_the_framework_strategy_bound_to_the_group_aggregator() -> None:
    plan, slots = _plan()
    root = _aggregator()
    child = _aggregator()
    s_root, s_child = plan.strategy_for("root", root), plan.strategy_for("team-a", child)
    assert isinstance(s_root, KappaSoftmaxAverage) and s_root.aggregator is root
    assert s_child.aggregator is child and s_child is not s_root
    assert not plan.customized() and plan.history() == []


def test_a_default_plan_builds_nothing_through_the_registry(monkeypatch) -> None:
    calls = {"build": 0}
    original = ComponentRegistry.build

    def counting(self, *args, **kwargs):
        calls["build"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ComponentRegistry, "build", counting)
    plan, _ = _plan()
    plan.strategy_for("root", _aggregator())
    plan.validate(VIEWS, initial=True)
    assert calls["build"] == 0


def test_an_in_code_kappa_instance_is_rebound_and_any_other_instance_is_shared(parts) -> None:
    shared = Picky(minimum=1)
    plan, _ = _plan(code_spec=shared)
    aggregator = _aggregator()
    assert plan.strategy_for("a", aggregator) is shared is plan.strategy_for("b", aggregator)
    assert plan.customized()

    class Cloning(Picky):
        def clone(self):
            return Cloning(minimum=self.minimum)

    plan, _ = _plan(code_spec=Cloning())
    assert plan.strategy_for("a", aggregator) is not plan.strategy_for("b", aggregator)

    plan, _ = _plan(code_spec=KappaSoftmaxAverage(_aggregator()))
    rebound = plan.strategy_for("a", aggregator)
    assert rebound.aggregator is aggregator
    assert not plan.customized()                       # a plain kappa instance is the default


# -- A3: the framework owns the bounds ------------------------------------------------------------------
def test_spec_cannot_set_floor() -> None:
    coopetition = {"WEIGHT_FLOOR": 0.02}
    with pytest.raises(ComponentSpecError, match="E011.*weight_floor.*supplied by the framework"):
        build_aggregator({"type": "default", "params": {"weight_floor": 0.4}}, coopetition, num_peers=2)
    with pytest.raises(ComponentSpecError, match="E011.*aggregator"):
        _plan({"WEIGHT_UPDATE": {"type": "kappa_softmax", "params": {"aggregator": object()}}})


def test_a_custom_strategy_that_declares_no_aggregator_is_not_offered_one(parts) -> None:
    plan, _ = _plan({"WEIGHT_UPDATE": "ema_softmax"})
    assert isinstance(plan.strategy_for("root", _aggregator()), EmaSoftmax)


# -- A9: fail at build ------------------------------------------------------------------------------------
def test_bad_spec_fails_at_build_unknown_name_and_key(parts) -> None:
    with pytest.raises(ComponentMissing, match="no_such"):
        _plan({"WEIGHT_UPDATE": "no_such"})
    with pytest.raises(ComponentParamError, match="alhpa"):
        _plan({"WEIGHT_UPDATE": {"type": "ema_softmax", "params": {"alhpa": 0.3}}})
    with pytest.raises(ValidationError):                                     # the legacy type is the same
        build_weight_update({"type": "uniform", "tolrance": 1})
    with pytest.raises(ComponentParamError):
        build_weight_update({"type": "score_weighted", "epsilon": 1e-3, "bogus": 1})


def test_bad_spec_fails_at_build_validate_probe_and_ambiguity(parts) -> None:
    plan, _ = _plan({"WEIGHT_UPDATE": {"type": "picky", "params": {"minimum": 4}}})
    with pytest.raises(ImproperlyConfigured, match="E064.*cannot serve group 'root'.*needs 4 peers"):
        plan.validate(VIEWS, initial=True)
    plan, _ = _plan({"WEIGHT_UPDATE": "sums_to_three"})
    with pytest.raises(ImproperlyConfigured, match="E064.*failed its probe.*U005"):
        plan.validate(VIEWS, initial=True)
    plan, _ = _plan({"WEIGHT_UPDATE": "raises"})
    with pytest.raises(ImproperlyConfigured, match="E064.*U007.*KeyError"):
        plan.validate(VIEWS, initial=True)
    plan, _ = _plan({"WEIGHT_UPDATE_OVERRIDES": {"te*": "uniform", "*-a": "score_weighted"}})
    with pytest.raises(ImproperlyConfigured, match="E061"):
        plan.validate(VIEWS, initial=True)


def test_an_override_that_matches_no_group_is_e060_at_build_and_w022_later(parts, caplog) -> None:
    plan, _ = _plan({"WEIGHT_UPDATE_OVERRIDES": {"nowhere": "uniform"}})
    with pytest.raises(ImproperlyConfigured, match="E060.*matches no aggregation group"):
        plan.validate(VIEWS, initial=True)
    with caplog.at_level(logging.WARNING, logger="mactf.weights"):
        notes = plan.validate(VIEWS, initial=False)
    assert notes and notes[0].startswith("W022")
    lax, _ = _plan({"WEIGHT_UPDATE_OVERRIDES": {"nowhere": "uniform"}, "WEIGHT_OPTIONS": {"STRICT_GLOBS": False}})
    assert lax.validate(VIEWS, initial=True)[0].startswith("W022")


def test_colliding_group_labels_are_e060(parts) -> None:
    plan, _ = _plan()
    clash = _views(("root", None, ("a", "b")), ("root", "root", ("root/x", "root/y")))
    with pytest.raises(ImproperlyConfigured, match="E060.*collide"):
        plan.validate(clash, initial=True)


@pytest.mark.parametrize("block,match", [
    ({"WEIGHT_OPTIONS": {"MIGRATE": "nope"}}, "E060.*MIGRATE"),
    ({"WEIGHT_OPTIONS": {"BAD": 1}}, "E060.*unknown keys"),
    ({"WEIGHT_GUARD": {"MODE": "nope"}}, "E060.*MODE"),
    ({"WEIGHT_UPDATE_OVERRIDES": ["uniform"]}, "E060.*mapping"),
    ({"WEIGHT_UPDATE_OVERRIDES": {" bad ": "uniform"}}, "E060.*glob"),
])
def test_e060_bad_blocks(block, match) -> None:
    with pytest.raises(ImproperlyConfigured, match=match):
        _plan(block)


# -- A6: atomic swap ------------------------------------------------------------------------------------------
def _armed(parts):
    plan, slots = _plan({"WEIGHT_UPDATE": "ema_softmax", "WEIGHT_UPDATE_OVERRIDES": {"team-a": "uniform"}})
    aggregator = _aggregator()
    vectors = {name: torch.zeros(2) for name in ("x", "y")}
    for label in ("root", "team-a", "team-b"):
        plan.strategy_for(label, aggregator).apply(vectors, {"x": 1.0, "y": 0.0}, kappa=1.0)
    return plan, slots, aggregator


@pytest.mark.parametrize("step,candidate,expected", [
    ("validate", {"type": "picky", "params": {"minimum": 9}}, "E064.*cannot serve"),
    ("probe", "sums_to_three", "E064.*probe"),
    ("probe_raises", "raises", "E064.*probe"),
    ("build_unknown", "no_such", "no_such"),
    ("build_param", {"type": "ema_softmax", "params": {"nope": 1}}, "nope"),
    ("migrate", "uniform", "E064.*migrate is 'fail'"),
    ("lock", "uniform", "locked"),
    ("commit", "uniform", "injected commit fault"),
])
def test_swap_atomic(parts, monkeypatch, step, candidate, expected) -> None:
    plan, slots, aggregator = _armed(parts)
    before = _fingerprint(plan, slots)
    state_before = plan.state_dict()
    kwargs = {}
    if step == "migrate":
        kwargs["mode"] = "fail"                         # team-b holds EMA state: the swap would drop it
    if step == "lock":
        monkeypatch.setattr(SlotTable, "_slot_locked", staticmethod(lambda path: True))
    if step == "commit":
        def boom(self, *args, **kw):
            raise RuntimeError("injected commit fault")
        monkeypatch.setattr(SlotTable, "swap", boom)
    with pytest.raises(Exception, match=expected):
        plan.set(None, candidate, views=VIEWS, boundary=Boundary.EPISODE, **kwargs)
    assert _fingerprint(plan, slots) == before         # slots, history, cache, carry, queue: untouched
    assert plan.state_dict() == state_before


def test_swap_atomic_boundary_and_adding_an_override(parts) -> None:
    plan, slots, aggregator = _armed(parts)
    before = _fingerprint(plan, slots)
    with pytest.raises(ComponentLocked, match="episode boundaries"):
        plan.set(None, "uniform", views=VIEWS, boundary=Boundary.STEP)
    with pytest.raises(ImproperlyConfigured, match="E060.*invalid override glob"):
        plan.set(" team ", "uniform", views=VIEWS)
    plan.set("te*", "uniform", views=VIEWS)                          # an addition: no record
    after_addition = _fingerprint(plan, slots)
    assert before != after_addition and "te*" in plan.override_globs()
    with pytest.raises(ImproperlyConfigured, match="E061"):          # team-b now matches two globs of one weight
        plan.set("*-b", "uniform", views=VIEWS)
    assert _fingerprint(plan, slots) == after_addition


def test_a_successful_swap_changes_only_the_affected_groups(parts) -> None:
    plan, slots, aggregator = _armed(parts)
    keep = plan.strategy_for("team-a", aggregator)                  # served by its own override
    ema_root = plan.strategy_for("root", aggregator)
    assert isinstance(ema_root, EmaSoftmax)
    record = plan.set(None, "score_weighted", views=VIEWS)
    assert record.to_name == "score_weighted" and record.from_name == "ema_softmax"
    assert plan.strategy_for("team-a", aggregator) is keep          # its slot did not change
    assert type(plan.strategy_for("root", aggregator)).__name__ == "ScoreWeightedAverage"
    assert plan.history()[-1].boundary == "episode"


def test_swap_mode_reset_drops_group_state_and_carry_keeps_it(parts, caplog) -> None:
    plan, slots, aggregator = _armed(parts)
    ema = plan.strategy_for("team-b", aggregator)._ema
    assert ema
    plan.set(None, {"type": "ema_softmax", "params": {"alpha": 0.9}}, views=VIEWS, mode="reset")
    assert plan.strategy_for("team-b", aggregator)._ema == {}
    plan2, _, aggregator2 = _armed(parts)
    ema2 = dict(plan2.strategy_for("team-b", aggregator2)._ema)
    plan2.set(None, {"type": "ema_softmax", "params": {"alpha": 0.9}}, views=VIEWS, mode="carry")
    carried = plan2.strategy_for("team-b", aggregator2)
    assert carried.alpha == 0.9 and carried._ema == ema2            # same class: state moved across
    plan3, _, aggregator3 = _armed(parts)
    plan3.set(None, "score_weighted", views=VIEWS, mode="carry")     # different class: nothing to carry
    assert not hasattr(plan3.strategy_for("team-b", aggregator3), "_ema")


def test_remove_an_override_returns_its_groups_to_the_next_rule(parts) -> None:
    plan, slots, aggregator = _armed(parts)
    assert type(plan.strategy_for("team-a", aggregator)).__name__ == "UniformAverage"
    record = plan.remove("team-a", views=VIEWS)
    assert record.to_name == "<removed>"
    assert isinstance(plan.strategy_for("team-a", aggregator), EmaSoftmax)
    with pytest.raises(ComponentMissing):
        plan.remove("team-a", views=VIEWS)


def test_remove_that_creates_a_tie_is_refused_and_changes_nothing(parts) -> None:
    plan, slots = _plan({"WEIGHT_UPDATE_OVERRIDES": {"abab": "uniform", "ab*": "score_weighted", "*ab": "none"}})
    views = _views(("root", None, ("p", "q")), ("abab", "abab", ("abab/x", "abab/y")))
    before = _fingerprint(plan, slots)
    with pytest.raises(ImproperlyConfigured, match="E061"):
        plan.remove("abab", views=views)
    assert _fingerprint(plan, slots) == before


def test_request_validates_now_and_applies_later(parts) -> None:
    plan, slots, aggregator = _armed(parts)
    with pytest.raises(ComponentMissing):
        plan.request(None, "no_such")                                 # fails on the request line
    with pytest.raises(ComponentParamError):
        plan.request(None, {"type": "ema_softmax", "params": {"x": 1}})
    plan.request(None, "uniform")
    plan.request_remove("team-a")
    assert plan.pending() == {DEFAULT_PATH: "set", OVERRIDE_PREFIX + "team-a": "remove"}
    assert isinstance(plan.strategy_for("root", aggregator), EmaSoftmax)        # nothing applied yet
    drained = plan.take_pending()
    assert [item[0] for item in drained] == [None, "team-a"] and plan.pending() == {}
    assert plan.is_removal(drained[1][1]) and not plan.is_removal(drained[0][1])
    with pytest.raises(ComponentMissing):
        plan.request_remove("never-was")


# -- A8: restore ------------------------------------------------------------------------------------------------
def test_resume_restores_slots_and_group_state_on_first_use(parts) -> None:
    plan, slots, aggregator = _armed(parts)
    saved = plan.state_dict()
    assert saved["version"] == 1 and set(saved["slots"]) == {DEFAULT_PATH, OVERRIDE_PREFIX + "team-a"}
    assert saved["groups"]["root"]["class"] == "EmaSoftmax" and saved["groups"]["root"]["state"]["ema"]

    fresh, fresh_slots = _plan({"WEIGHT_UPDATE": "ema_softmax", "WEIGHT_UPDATE_OVERRIDES": {"team-a": "uniform"}})
    notes = fresh.load_state_dict(saved)
    assert notes == []                                              # same specs: nothing to report
    restored = fresh.strategy_for("root", _aggregator())
    assert restored._ema == saved["groups"]["root"]["state"]["ema"]
    assert fresh.state_dict()["groups"]["root"] == saved["groups"]["root"]


def test_resume_a_changed_default_is_w021_and_the_saved_plan_wins(parts) -> None:
    plan, _, _ = _armed(parts)
    saved = plan.state_dict()
    other, _ = _plan({"WEIGHT_UPDATE": {"type": "ema_softmax", "params": {"alpha": 0.9}},
                      "WEIGHT_UPDATE_OVERRIDES": {"team-a": "uniform"}})
    notes = other.load_state_dict(saved)
    assert any(note.startswith("W021") and DEFAULT_PATH in note for note in notes)
    assert other.strategy_for("root", _aggregator()).alpha == 0.3        # the saved plan won


def test_resume_strict_restore_is_e066(parts) -> None:
    plan, _, _ = _armed(parts)
    other, _ = _plan({"WEIGHT_UPDATE": "uniform"}, components={"STRICT_RESTORE": True})
    with pytest.raises(ImproperlyConfigured, match="E066"):
        other.load_state_dict(plan.state_dict(), strict=True)


def test_resume_a_removed_override_and_a_class_change_drop_state(parts, caplog) -> None:
    plan, _, _ = _armed(parts)
    saved = plan.state_dict()
    renamed = copy.deepcopy(saved)
    renamed["groups"]["root"]["class"] = "SomethingElse"
    fresh, _ = _plan({"WEIGHT_UPDATE": "ema_softmax"})            # the override is gone from this config
    with caplog.at_level(logging.WARNING, logger="mactf.weights"):
        notes = fresh.load_state_dict(renamed)
    assert any("W021" in note and "team-a" in note for note in notes)          # the saved override returns
    assert fresh.override_globs() == ["team-a"]
    assert fresh.strategy_for("root", _aggregator())._ema == {}                # state for another class: dropped
    assert any("does not fit" in record.message for record in caplog.records)


def test_resume_an_in_code_instance_is_never_rebuilt(parts) -> None:
    plan, _ = _plan(code_spec=Picky(minimum=1))
    saved = plan.state_dict()
    assert "adopted" in saved["slots"][DEFAULT_PATH]
    fresh, _ = _plan({"WEIGHT_UPDATE": "uniform"})
    notes = fresh.load_state_dict(saved)
    assert any("in-code instance" in note for note in notes)
    assert isinstance(fresh.strategy_for("root", _aggregator()), UniformAverage)


# -- KappaPlan -------------------------------------------------------------------------------------------------
def test_kappa_plan_applies_the_most_specific_override() -> None:
    plan = KappaPlan({"root": {"ETA_KAPPA": 0.02}, "team-*": {"KAPPA_INIT": 3.0, "TARGET_ENTROPY_RATIO": 0.95}})
    base = {"KAPPA_INIT": 1.0, "ETA_KAPPA": 0.05}
    assert plan.settings_for("root", base)["ETA_KAPPA"] == 0.02
    assert plan.settings_for("team-a", base)["KAPPA_INIT"] == 3.0
    assert plan.settings_for("other", base) == base
    controller = plan.build("team-a", {"KAPPA_INIT": 1.0}, num_peers=3)
    assert controller.kappa == 3.0 and controller.target_ratio == 0.95 and controller.num_peers == 3


@pytest.mark.parametrize("overrides,match", [
    ({"root": {"WEIGHT_FLOOR": 0.1}}, "may not set WEIGHT_FLOOR"),
    ({"root": {"WEIGHT_CAP": 0.5}}, "may not set WEIGHT_CAP"),
    ({"root": {"SIGNAL_SOURCE": "points"}}, "may not set SIGNAL_SOURCE"),
    ({"root": {"NOPE": 1}}, "unknown key NOPE"),
    ({"root": {"KAPPA_INIT": float("nan")}}, "finite number"),
    ({"root": {"KAPPA_INIT": True}}, "finite number"),
    ({"root": {}}, "non-empty mapping"),
    ({"root": "x"}, "non-empty mapping"),
    ({" root": {"KAPPA_INIT": 1}}, "invalid glob"),
])
def test_e062_bad_kappa_overrides(overrides, match) -> None:
    with pytest.raises(ImproperlyConfigured, match=f"E062.*{match}"):
        KappaPlan(overrides)


def test_e062_bad_merged_bounds_and_ties_and_unmatched_globs() -> None:
    with pytest.raises(ImproperlyConfigured, match="E062.*KAPPA_MIN <= KAPPA_MAX"):
        KappaPlan({"root": {"KAPPA_MIN": 5.0}}).settings_for("root", {"KAPPA_MAX": 2.0})
    with pytest.raises(ImproperlyConfigured, match="E062.*TARGET_ENTROPY_RATIO"):
        KappaPlan({"root": {"TARGET_ENTROPY_RATIO": 1.5}}).settings_for("root", {})
    with pytest.raises(ImproperlyConfigured, match="E062.*BETA_MOMENTUM"):
        KappaPlan({"root": {"BETA_MOMENTUM": 1.0}}).settings_for("root", {})
    with pytest.raises(ImproperlyConfigured, match="E062.*equally specific"):
        KappaPlan({"ab*": {"KAPPA_INIT": 2}, "*ab": {"KAPPA_INIT": 3}}).settings_for("abab", {})
    plan = KappaPlan({"nowhere": {"KAPPA_INIT": 2}})
    with pytest.raises(ImproperlyConfigured, match="E062.*matches no aggregation group"):
        plan.validate(["root"], initial=True)
    assert plan.validate(["root"], initial=False)[0].startswith("W022")
    assert KappaPlan({"nowhere": {"KAPPA_INIT": 2}}, strict_globs=False).validate(["root"], initial=True)


def test_kappa_plan_from_settings_reads_the_coopetition_block() -> None:
    assert KappaPlan.from_settings({}).overrides == {}
    assert KappaPlan.from_settings({"KAPPA_OVERRIDES": {"root": {"eta_kappa": 0.1}}}).overrides == {"root": {"ETA_KAPPA": 0.1}}
    with pytest.raises(ImproperlyConfigured, match="E062"):
        KappaPlan.from_settings({"KAPPA_OVERRIDES": [1]})
    assert set(KAPPA_KEYS) == {"KAPPA_INIT", "KAPPA_MIN", "KAPPA_MAX", "TARGET_ENTROPY_RATIO",
                               "BETA_MOMENTUM", "ETA_KAPPA"}


# -- the temperature fixes (W-G7, A7) ----------------------------------------------------------------------------
def test_snapshot_restore_undoes_a_step_exactly() -> None:
    controller = AdaptiveKappa(num_peers=3)
    before = controller.snapshot()
    state_object = controller.state
    controller.step(torch.tensor([3.0, 1.0, -2.0]))
    assert controller.state.step == 1
    controller.restore(before)
    assert controller.state is state_object                         # restored in place
    assert controller.state.step == 0 and controller.state.history == [] and controller.kappa == 1.0
    assert controller.snapshot()["state"] == before["state"]


def test_resize_retargets_the_entropy_and_load_restores_the_arity() -> None:
    controller = AdaptiveKappa(num_peers=4)
    controller.resize(2)
    assert controller.num_peers == 2 and controller.target_entropy == pytest.approx(0.8 * 0.6931471805599453)
    with pytest.raises(ValueError):
        controller.resize(0)
    saved = controller.state_dict()
    fresh = AdaptiveKappa(num_peers=9)
    fresh.load_state_dict(saved)
    assert fresh.num_peers == 2 and fresh.target_entropy == controller.target_entropy
    legacy = {k: v for k, v in saved.items() if k != "num_peers"}
    again = AdaptiveKappa(num_peers=9)
    again.load_state_dict(legacy)                                  # an older checkpoint: as before
    assert again.num_peers == 9


# -- the aggregator ----------------------------------------------------------------------------------------------
class TrackedAggregator(Aggregator):
    def __init__(self, *, tag: str = "t", **kwargs) -> None:
        super().__init__(**kwargs)
        self.tag = tag

    def options(self):
        return {**super().options(), "tag": self.tag}


class ForgetfulAggregator(Aggregator):
    def __init__(self, *, tag: str, **kwargs) -> None:             # a required argument ...
        super().__init__(**kwargs)
        self.tag = tag                                              # ... and options() not overridden


def test_clone_with_pristine_and_options_round_trip() -> None:
    base = Aggregator(controller=AdaptiveKappa(num_peers=2), weight_floor=0.05, trim_fraction=0.1)
    sibling = base.clone_with(AdaptiveKappa(num_peers=3))
    assert type(sibling) is Aggregator and sibling.options() == base.options()
    assert sibling.controller is not base.controller
    tracked = TrackedAggregator(controller=AdaptiveKappa(num_peers=2), tag="x", weight_cap=0.5)
    clone = tracked.clone_with(AdaptiveKappa(num_peers=2))
    assert type(clone) is TrackedAggregator and clone.tag == "x" and clone.weight_cap == 0.5
    plain = tracked.pristine()
    assert type(plain) is Aggregator and plain.controller is tracked.controller and plain.weight_cap == 0.5


def test_verify_aggregator_e065_for_a_subclass_that_cannot_be_cloned() -> None:
    with pytest.raises(ImproperlyConfigured, match="E065.*clone_with"):
        verify_aggregator(ForgetfulAggregator(controller=AdaptiveKappa(num_peers=2), tag="z"))
    verify_aggregator(TrackedAggregator(controller=AdaptiveKappa(num_peers=2)))
    verify_aggregator(Aggregator(controller=AdaptiveKappa(num_peers=2)))


def test_build_aggregator_injects_the_bounds_and_the_controller(component_overlay) -> None:
    register_aggregator("tracked", TrackedAggregator, version="1", scope="run")
    coop = {"WEIGHT_FLOOR": 0.03, "WEIGHT_CAP": 0.55, "KAPPA_INIT": 2.0}
    built = build_aggregator({"type": "tracked", "params": {"tag": "q"}}, coop, num_peers=3)
    assert isinstance(built, TrackedAggregator) and built.tag == "q"
    assert (built.weight_floor, built.weight_cap) == (0.03, 0.55) and built.controller.kappa == 2.0
    register_aggregator("forgetful", ForgetfulAggregator, version="1", scope="run")
    with pytest.raises(ImproperlyConfigured, match="E065"):
        build_aggregator({"type": "forgetful", "params": {"tag": "q"}}, coop, num_peers=3)
    plain = build_aggregator(None, coop, num_peers=3)
    assert type(plain) is Aggregator and plain.options() == Aggregator.from_settings(coop, num_peers=3).options()
    existing = Aggregator(controller=AdaptiveKappa(num_peers=2))
    assert build_aggregator(existing, coop, num_peers=3) is existing


# -- the trainer ------------------------------------------------------------------------------------------------------
def _flat(settings, tmp_path, block=None, *, peers=3, epochs=2, **kwargs):
    tune(settings, EPOCHS=epochs)
    if block is not None:
        configure(settings, TRAINER=block)
    return make_trainer(settings, tmp_path, peers=peers, **kwargs)


def _hier(settings, tmp_path, block=None, *, epochs=2, **kwargs):
    tune(settings, EPOCHS=epochs)
    if block is not None:
        configure(settings, TRAINER=block)
    return hierarchy_trainer(settings, tmp_path, **kwargs)


def test_weights_default_identity(settings, tmp_path) -> None:
    """A1: with nothing configured the strategies, controllers and reports are today's."""
    trainer = _flat(settings, tmp_path, peers=3).build()
    assert type(trainer.weight_strategy) is KappaSoftmaxAverage
    assert trainer.weight_strategy.aggregator is trainer.aggregator
    assert type(trainer.aggregator) is Aggregator and trainer.controllers == {"root": trainer.controller}
    assert not trainer.weight_guard.covers(trainer.weight_strategy, trainer.aggregator)
    assert not trainer.weight_plan.customized() and trainer.weight_warnings == []

    oracle = Aggregator.from_settings(dict(settings["COOPETITION"]), num_peers=3)
    trainer.run_epoch(0)
    report = trainer.reports[0]
    names = sorted(trainer.tree.root_paths)
    eligible = [not trainer._void(name) for name in names]
    expected = oracle.compute_weights(
        names, [float(trainer.signal_by_peer.get(name, 0.0)) for name in names], eligible=eligible,
        confidence=[trainer._confidence(name) for name in names],
    )
    assert report.weights == expected.weights                       # bit for bit
    assert (report.kappa, report.entropy, report.raw_weights) == (expected.kappa, expected.entropy, expected.raw_weights)
    payload = trainer._trainer_state()
    assert "weight_plan" not in payload and "strategy_plan" not in payload and "freeze" not in payload


def test_weights_default_identity_in_a_hierarchy(settings, tmp_path) -> None:
    trainer = _hier(settings, tmp_path).build()
    assert set(trainer.controllers) == {"root", "peer-1", "peer-2"}
    trainer.run_epoch(0)
    groups = {tuple(report.order) for report in trainer.reports}
    assert groups == {("peer-1", "peer-2"), ("peer-1/sub-1", "peer-1/sub-2"), ("peer-2/sub-1", "peer-2/sub-2")}
    for parent in ("peer-1", "peer-2"):
        aggregator = trainer._group_aggregator(parent)
        assert type(aggregator) is Aggregator and aggregator.controller is trainer.controllers[parent]
        assert aggregator.options() == trainer.aggregator.options()
        assert isinstance(trainer.weight_plan.strategy_for(parent, aggregator), KappaSoftmaxAverage)


def test_trainer_weight_update_setting_now_drives_behaviour(settings, tmp_path) -> None:
    trainer = _flat(settings, tmp_path, {"WEIGHT_UPDATE": "uniform"}, peers=3).build()
    assert trainer.weight_strategy.name == "uniform" and trainer.spec.weight_update == "uniform"
    trainer.run_epoch(0)
    assert list(trainer.reports[0].weights.values()) == pytest.approx([1 / 3] * 3)
    # and the constructor argument still beats the setting
    other = _flat(settings, tmp_path / "b", {"WEIGHT_UPDATE": "uniform"}, peers=3, weight_update="score_weighted").build()
    assert other.weight_strategy.name == "score_weighted"


def test_per_group_strategies_and_state_in_a_hierarchy(settings, tmp_path, parts) -> None:
    block = {"WEIGHT_UPDATE": "ema_softmax", "WEIGHT_UPDATE_OVERRIDES": {"peer-1": "uniform"}}
    trainer = _hier(settings, tmp_path, block).build()
    trainer.run_epoch(0)
    by_order = {tuple(report.order): report for report in trainer.reports}
    assert list(by_order[("peer-1/sub-1", "peer-1/sub-2")].weights.values()) == pytest.approx([0.5, 0.5])
    root = trainer.weight_plan.strategy_for("root", trainer.aggregator)
    sub = trainer.weight_plan.strategy_for("peer-2", trainer._group_aggregator("peer-2"))
    assert isinstance(root, EmaSoftmax) and isinstance(sub, EmaSoftmax) and root is not sub
    assert set(root._ema) == {"peer-1", "peer-2"} and set(sub._ema) == {"peer-2/sub-1", "peer-2/sub-2"}


def test_a_validate_failure_stops_the_build(settings, tmp_path, parts) -> None:
    trainer = _flat(settings, tmp_path, {"WEIGHT_UPDATE": {"type": "picky", "params": {"minimum": 5}}}, peers=3)
    with pytest.raises(ImproperlyConfigured, match="E064"):
        trainer.build()


def test_reference_only_is_no_longer_masked(settings, tmp_path) -> None:
    trainer = _flat(settings, tmp_path, {"WEIGHT_UPDATE": {"type": "reference_only", "params": {"reference": "peer-2"}}}, peers=3).build()
    trainer.run_epoch(0)
    report = trainer.reports[0]
    assert report.fallback and report.weights["peer-2"] == 1.0
    assert not any("fell back" in note for note in report.adjustments)


@pytest.mark.parametrize("name", ["sums_to_three", "raises"])
def test_a_bad_strategy_cannot_reach_the_aggregator(settings, tmp_path, parts, name) -> None:
    """With the probe off (so the build accepts it) the guard still keeps the weights legal."""
    block = {"WEIGHT_UPDATE": name, "WEIGHT_GUARD": {"PROBE": False}}
    trainer = _flat(settings, tmp_path, block, peers=3).build()
    trainer.run_epoch(0)
    report = trainer.reports[0]
    assert sum(report.weights.values()) == pytest.approx(1.0, abs=1e-5)
    assert all(0.0 <= value <= 0.6 + 1e-9 for value in report.weights.values())
    for peer in trainer.peers.values():
        assert all(torch.isfinite(p).all() for p in peer.adapter.module.parameters())
    assert any(name in note or "weight_guard" in note for note in report.adjustments)


def test_guard_modes_in_a_run(settings, tmp_path, parts) -> None:
    repaired = _flat(settings, tmp_path / "r", {"WEIGHT_UPDATE": "sums_to_three", "WEIGHT_GUARD": {"PROBE": False}}).build()
    repaired.run_epoch(0)
    assert any("weight_guard repaired weights" in note for note in repaired.reports[0].adjustments)
    fallback = _flat(settings, tmp_path / "f", {"WEIGHT_UPDATE": "sums_to_three",
                                                "WEIGHT_GUARD": {"PROBE": False, "MODE": "fallback"}}).build()
    fallback.run_epoch(0)
    assert any("fell back to kappa_softmax" in note for note in fallback.reports[0].adjustments)
    strict = _flat(settings, tmp_path / "e", {"WEIGHT_UPDATE": "sums_to_three",
                                              "WEIGHT_GUARD": {"PROBE": False, "MODE": "enforce"}}).build()
    with pytest.raises(WeightGuardViolation, match="E063"):
        strict.run_epoch(0)


def test_probe_enforce_catches_it_at_build(settings, tmp_path, parts) -> None:
    trainer = _flat(settings, tmp_path, {"WEIGHT_UPDATE": "sums_to_three"})
    with pytest.raises(ImproperlyConfigured, match="E064.*probe"):
        trainer.build()


def test_fallback_does_not_double_step(settings, tmp_path, parts) -> None:
    """A7: a rejected proposal restores the controller before the fallback runs."""
    class AdvancingFactory(Advancing):
        pass

    trainer = _flat(settings, tmp_path, {"WEIGHT_GUARD": {"PROBE": False}}, peers=3,
                    weight_update=Advancing(None)).build()
    # the in-code instance is shared; bind it to the run's aggregator so it advances the real controller
    trainer.weight_plan._adopted[DEFAULT_PATH].aggregator = trainer.aggregator
    trainer.run_epoch(0)
    assert trainer.controller.state.step == 1                      # once for the fallback, not twice
    report = trainer.reports[0]
    assert any("fell back to kappa_softmax" in note or "weight_guard" in note for note in report.adjustments)


def test_a_raising_strategy_does_not_end_the_run(settings, tmp_path, parts) -> None:
    trainer = _flat(settings, tmp_path, {"WEIGHT_UPDATE": "raises", "WEIGHT_GUARD": {"PROBE": False}}, peers=3).build()
    result = trainer.fit(epochs=2)
    assert result.status == "completed"
    assert all("fell back to kappa_softmax" in " ".join(r.adjustments) for r in trainer.reports)


def test_kappa_overrides_build_per_group_controllers(settings, tmp_path) -> None:
    configure(settings, COOPETITION={"KAPPA_OVERRIDES": {"root": {"KAPPA_INIT": 2.0}, "peer-*": {"KAPPA_INIT": 4.0, "ETA_KAPPA": 0.2}}})
    trainer = _hier(settings, tmp_path).build()
    assert trainer.controllers["root"].kappa == 2.0 and trainer.aggregator.controller.kappa == 2.0
    assert trainer.controllers["peer-1"].kappa == 4.0 and trainer.controllers["peer-1"].eta_kappa == 0.2
    assert trainer._group_aggregator("peer-2").controller is trainer.controllers["peer-2"]
    configure(settings, COOPETITION={"KAPPA_OVERRIDES": {"nowhere": {"KAPPA_INIT": 4.0}}})
    with pytest.raises(ImproperlyConfigured, match="E062"):
        _hier(settings, tmp_path / "b").build()


def test_a_custom_aggregator_reaches_every_group(settings, tmp_path, component_overlay) -> None:
    register_aggregator("tracked", TrackedAggregator, version="1", scope="run")
    trainer = _hier(settings, tmp_path, {"AGGREGATOR": {"type": "tracked", "params": {"tag": "k"}}}).build()
    assert isinstance(trainer.aggregator, TrackedAggregator)
    child = trainer._group_aggregator("peer-1")
    assert type(child) is TrackedAggregator and child.tag == "k"              # W-G9: no longer downcast
    trainer.run_epoch(0)


def test_live_swap_through_the_trainer(settings, tmp_path, parts) -> None:
    trainer = _hier(settings, tmp_path, {"WEIGHT_UPDATE": "ema_softmax"}).build()
    trainer.run_epoch(0)
    trainer.request_weight_update("peer-*", "uniform")
    with pytest.raises(ComponentMissing):
        trainer.request_weight_update(None, "no_such")
    record = trainer.swap_weight_update("peer-1", "score_weighted")
    assert record is None                                          # an addition records no swap
    assert trainer.weight_plan.override_globs() == ["peer-1"]
    trainer.run_epoch(1)                                           # the queued override lands at the boundary
    assert trainer.weight_plan.override_globs() == ["peer-*", "peer-1"]
    assert trainer.failures == []
    trainer.remove_weight_override("peer-*")
    assert trainer.weight_plan.override_globs() == ["peer-1"]
    with pytest.raises(ComponentLocked):
        trainer.swap_weight_update(None, "uniform", boundary=Boundary.STEP)


def test_a_failed_queued_weight_change_is_recorded_not_raised(settings, tmp_path, parts) -> None:
    trainer = _flat(settings, tmp_path, {"WEIGHT_UPDATE": "ema_softmax"}, peers=3).build()
    trainer.request_weight_update(None, {"type": "picky", "params": {"minimum": 9}})
    trainer.run_epoch(0)                                           # the boundary refuses it (E064), the run goes on
    assert any("queued weight_update change" in failure and "E064" in failure for failure in trainer.failures)
    assert isinstance(trainer.weight_plan.strategy_for("root", trainer.aggregator), EmaSoftmax)


def test_resume_a_weight_plan_through_the_trainer(settings, tmp_path, parts) -> None:
    block = {"WEIGHT_UPDATE": {"type": "ema_softmax", "params": {"alpha": 0.5}}}
    trainer = _flat(settings, tmp_path / "a", block, peers=3).build()
    trainer.run_epoch(0)
    trainer.run_epoch(1)
    state = trainer._trainer_state()
    assert state["weight_plan"]["groups"]["root"]["class"] == "EmaSoftmax"
    ema = trainer.weight_plan.strategy_for("root", trainer.aggregator)._ema
    payload = trainer.state_dict()
    fresh = _flat(settings, tmp_path / "b", block, peers=3)
    fresh.run_id = trainer.run_id
    fresh.build()
    fresh.load_state_dict(payload)
    assert fresh.weight_plan.strategy_for("root", fresh.aggregator)._ema == ema
    assert fresh.controller.state_dict() == trainer.controller.state_dict()


def test_a_stored_temperature_with_no_live_group_is_dropped_not_loaded_into_root(settings, tmp_path, caplog) -> None:
    """W-G10: the old restore sent an unknown label to the root controller."""
    trainer = _flat(settings, tmp_path / "a", peers=2).build()
    trainer.run_epoch(0)
    payload = trainer.state_dict()
    root_state = copy.deepcopy(payload["kappa"]["root"])
    payload["kappa"]["ghost-group"] = {**root_state, "step": 99, "kappa": 7.5}     # its group was pruned
    fresh = _flat(settings, tmp_path / "b", peers=2)
    fresh.run_id = trainer.run_id
    fresh.build()
    with caplog.at_level(logging.WARNING, logger="mactf.checkpoint"):
        report = fresh.load_state_dict(payload)
    assert "kappa:ghost-group:dropped" in report.restored_sections and "kappa:root" in report.restored_sections
    assert any("W021" in record.message and "ghost-group" in record.message for record in caplog.records)
    assert fresh.controller.state.step == root_state["step"]       # the root keeps its own state ...
    assert fresh.controller.kappa == root_state["kappa"] != 7.5    # ... not the ghost's


def test_a_default_trainer_writes_no_new_checkpoint_keys(settings, tmp_path) -> None:
    trainer = _flat(settings, tmp_path, peers=2).build()
    trainer.run_epoch(0)
    assert set(trainer._trainer_state()) == {"spec", "strategy", "callbacks", "history", "epoch", "signal_source",
                                             "distillation"} | ({"policy"} & set(trainer._trainer_state()))


def test_a_candidate_that_cannot_take_the_carried_state_changes_nothing(parts) -> None:
    class Strict(EmaSoftmax):
        name = "strict_load"

        def load_state_dict(self, payload):
            raise ValueError("wrong shape")

    register_weight_update("strict_load", Strict, version="1", scope="run")
    plan, slots = _plan({"WEIGHT_UPDATE": "strict_load", "WEIGHT_GUARD": {"PROBE": False}})
    aggregator = _aggregator()
    plan.strategy_for("team-a", aggregator).apply(
        {n: torch.zeros(2) for n in ("x", "y")}, {"x": 1.0, "y": 0.0}, kappa=1.0)       # team-a now holds state
    before = _fingerprint(plan, slots)
    with pytest.raises(ImproperlyConfigured, match="E064.*cannot take the carried state of group 'team-a'.*nothing was changed"):
        plan.set(None, "strict_load", views=VIEWS_TEAM, mode="carry")
    assert _fingerprint(plan, slots) == before
    plan.set(None, "strict_load", views=VIEWS_TEAM, mode="reset")               # reset never loads: allowed
    assert isinstance(plan.strategy_for("team-a", aggregator), Strict)


def test_a_stale_group_state_that_cannot_load_is_dropped_not_fatal(parts, caplog) -> None:
    class Strict(EmaSoftmax):
        name = "strict_load"

        def load_state_dict(self, payload):
            raise ValueError("wrong shape")

    register_weight_update("strict_load", Strict, version="1", scope="run")
    plan, _ = _plan({"WEIGHT_UPDATE": "strict_load"})
    plan._restore["root"] = {"slot": DEFAULT_PATH, "class": "Strict", "state": {"ema": {"x": 1.0}}}
    with caplog.at_level(logging.WARNING, logger="mactf.weights"):
        assert isinstance(plan.strategy_for("root", _aggregator()), Strict)       # does not raise
    assert any("could not be loaded" in record.message for record in caplog.records)


VIEWS_TEAM = _views(("root", None, ("team-a", "team-b")), ("team-a", "team-a", ("x", "y")))
