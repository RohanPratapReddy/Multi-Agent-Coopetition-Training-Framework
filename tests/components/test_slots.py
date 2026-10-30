"""Slots and swaps (document 17 section 4.4): the boundary rule, atomicity, the queue."""

from __future__ import annotations

import pytest

from component_helpers import OtherPolicy, Policy, Term
from mactf.components import (
    HUB,
    Boundary,
    ComponentResolver,
    ComponentSpec,
    Mutability,
    SlotTable,
    SwapRecord,
)
from mactf.exceptions import (
    ComponentBuildError,
    ComponentLocked,
    ComponentMigrationError,
    ComponentMissing,
    ComponentParamError,
)

POLICY = "trainer_policy"  # static
TERM = "reward_term"  # episode


class StepTerm(Term):
    pass


@pytest.fixture
def scope(component_overlay):
    """Registered components of every mutability class, in a run overlay."""
    reg = HUB.registry
    reg(POLICY).register("sl_static_a", Policy, scope="run")
    reg(POLICY).register("sl_static_b", OtherPolicy, scope="run")
    reg(TERM).register("sl_episode_a", Term, scope="run")
    reg(TERM).register("sl_episode_b", StepTerm, scope="run")
    reg(TERM).register("sl_static_term", Term, scope="run", mutability=Mutability.STATIC)
    # a step-class kind: rl gradient functions (default mutability step)
    reg("rl_function.gradient").register("sl_step_a", _Grad, scope="run")
    reg("rl_function.gradient").register("sl_step_b", _Grad2, scope="run")
    reg("rl_function.gradient").register("sl_step_ep", _Grad, scope="run", mutability=Mutability.EPISODE)
    return component_overlay


class _Grad:
    def __call__(self, *args):  # pragma: no cover
        return None


class _Grad2(_Grad):
    pass


@pytest.fixture
def slots(scope):
    return SlotTable(ComponentResolver())


# (slot kind, old, new, boundary) -> allowed; the rule is rank(boundary) <= min(rank(old), rank(new))
CASES = [
    (POLICY, "sl_static_a", "sl_static_b", Boundary.RUN_START, True),
    (POLICY, "sl_static_a", "sl_static_b", Boundary.EPISODE, False),
    (POLICY, "sl_static_a", "sl_static_b", Boundary.STEP, False),
    (TERM, "sl_episode_a", "sl_episode_b", Boundary.RUN_START, True),
    (TERM, "sl_episode_a", "sl_episode_b", Boundary.EPISODE, True),
    (TERM, "sl_episode_a", "sl_episode_b", Boundary.STEP, False),
    (TERM, "sl_episode_a", "sl_static_term", Boundary.EPISODE, False),  # the stricter of the two rules
    (TERM, "sl_static_term", "sl_episode_a", Boundary.EPISODE, False),
    (TERM, "sl_static_term", "sl_episode_a", Boundary.RUN_START, True),
    ("rl_function.gradient", "sl_step_a", "sl_step_b", Boundary.STEP, True),
    ("rl_function.gradient", "sl_step_a", "sl_step_b", Boundary.EPISODE, True),
    ("rl_function.gradient", "sl_step_a", "sl_step_b", Boundary.RUN_START, True),
    # an episode component can never be swapped between steps even if the replacement is step
    ("rl_function.gradient", "sl_step_a", "sl_step_ep", Boundary.STEP, False),
    ("rl_function.gradient", "sl_step_ep", "sl_step_a", Boundary.STEP, False),
    ("rl_function.gradient", "sl_step_ep", "sl_step_a", Boundary.EPISODE, True),
]


@pytest.mark.parametrize("kind, old, new, boundary, allowed", CASES,
                         ids=[f"{c[1]}->{c[2]}@{c[3].value}" for c in CASES])
def test_a_slot_changes_only_at_its_boundary(slots, kind, old, new, boundary, allowed):
    slots.install("SLOT", kind, old)
    if allowed:
        record = slots.swap("SLOT", new, boundary=boundary)
        assert slots.get("SLOT").component.name == new
        assert (record.from_name, record.to_name, record.boundary) == (old, new, boundary.value)
    else:
        with pytest.raises(ComponentLocked, match="may only change at"):
            slots.swap("SLOT", new, boundary=boundary)
        assert slots.get("SLOT").component.name == old  # nothing changed
        assert slots.history == ()


def test_the_message_names_the_boundary_that_is_needed(slots):
    slots.install("S", POLICY, "sl_static_a")
    with pytest.raises(ComponentLocked, match="may only change at run_start boundaries, not episode"):
        slots.swap("S", "sl_static_b", boundary=Boundary.EPISODE)
    slots.install("T", TERM, "sl_episode_a")
    with pytest.raises(ComponentLocked, match="may only change at episode boundaries, not step"):
        slots.swap("T", "sl_episode_b", boundary=Boundary.STEP)


def test_install_get_paths_match(slots):
    slots.install("TRAINER.POLICY", POLICY, "sl_static_a")
    slots.install("SCORING.REWARD_TERMS.R1", TERM, {"type": "sl_episode_a", "scale": 2.0})
    slots.install("SCORING.REWARD_TERMS.R2", TERM, "sl_episode_b")
    assert slots.paths() == ["SCORING.REWARD_TERMS.R1", "SCORING.REWARD_TERMS.R2", "TRAINER.POLICY"]
    assert [s.path for s in slots.match("SCORING.REWARD_TERMS.*")] == [
        "SCORING.REWARD_TERMS.R1", "SCORING.REWARD_TERMS.R2"]
    assert slots.match("NOPE.*") == []
    slot = slots.get("SCORING.REWARD_TERMS.R1")
    assert slot.instance.scale == 2.0 and slot.kind_id == TERM
    assert slot.component.name == "sl_episode_a" and slot.resolved.spec.params == {"scale": 2.0}


def test_a_missing_slot_is_e010_listing_the_paths(slots):
    slots.install("A", POLICY, "sl_static_a")
    with pytest.raises(ComponentMissing) as caught:
        slots.get("B")
    assert caught.value.code == "E010" and caught.value.known == ("A",)
    for call in (
        lambda: slots.swap("B", "x", boundary=Boundary.RUN_START),
        lambda: slots.request("B", "x"),
    ):
        with pytest.raises(ComponentMissing):
            call()


def test_adopt_records_an_instance_and_blocks_episode_swaps_for_a_static_kind(slots):
    instance = Policy(rate=9.0)
    slot = slots.adopt("TRAINER.POLICY", POLICY, instance)
    assert slot.instance is instance and slot.component.origin.value == "runtime"
    with pytest.raises(ComponentLocked):
        slots.swap("TRAINER.POLICY", "sl_static_a", boundary=Boundary.EPISODE)
    slots.swap("TRAINER.POLICY", "sl_static_a", boundary=Boundary.RUN_START)
    assert isinstance(slots.get("TRAINER.POLICY").instance, Policy)
    assert slots.get("TRAINER.POLICY").instance is not instance


# -- atomicity: a failed swap changes nothing ------------------------------------------------
def _state(slots):
    slot = slots.get("T")
    return (slot.component.name, slot.instance, slot.resolved.spec_hash, slots.history, slots.pending())


@pytest.mark.parametrize(
    "raw, error",
    [
        ("sl_missing", ComponentMissing),
        ({"type": "sl_episode_b", "bogus": 1}, ComponentParamError),
        ({"params": {"x": 1}}, Exception),  # a patch is not a component
        (7, Exception),
    ],
)
def test_a_failed_swap_leaves_the_old_component_running(slots, raw, error):
    slots.install("T", TERM, "sl_episode_a")
    before = _state(slots)
    with pytest.raises(error):
        slots.swap("T", raw, boundary=Boundary.EPISODE)
    assert _state(slots) == before


def test_a_failing_factory_leaves_the_old_component_running(slots):
    class Raises:
        rule_ids = frozenset({"R"})

        def __init__(self):
            raise ValueError("no")

        def contribution(self):  # pragma: no cover
            pass

    HUB.registry(TERM).register("sl_raises", Raises, scope="run")
    slots.install("T", TERM, "sl_episode_a")
    before = _state(slots)
    with pytest.raises(ComponentBuildError):
        slots.swap("T", "sl_raises", boundary=Boundary.EPISODE)
    assert _state(slots) == before


def test_migrate_runs_before_the_commit_and_sees_both_instances(slots):
    slots.install("T", TERM, "sl_episode_a")
    old = slots.get("T").instance
    seen = []

    def migrate(old_instance, new_instance):
        seen.append((old_instance, new_instance, slots.get("T").instance))  # slot still old here

    slots.swap("T", "sl_episode_b", boundary=Boundary.EPISODE, migrate=migrate)
    assert seen[0][0] is old and isinstance(seen[0][1], StepTerm) and seen[0][2] is old
    assert slots.get("T").instance is seen[0][1]


def test_a_failing_migration_is_e048_and_the_old_component_keeps_running(slots):
    slots.install("T", TERM, "sl_episode_a")
    before = _state(slots)

    def migrate(old, new):
        raise RuntimeError("cannot carry state")

    with pytest.raises(ComponentMigrationError) as caught:
        slots.swap("T", "sl_episode_b", boundary=Boundary.EPISODE, migrate=migrate)
    assert caught.value.code == "E048" and "cannot carry state" in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert _state(slots) == before


def test_a_component_error_from_migrate_propagates_unchanged(slots):
    slots.install("T", TERM, "sl_episode_a")

    def migrate(old, new):
        raise ComponentLocked("optimizer", "x", "no")

    with pytest.raises(ComponentLocked):
        slots.swap("T", "sl_episode_b", boundary=Boundary.EPISODE, migrate=migrate)


def test_the_swap_record_is_complete_and_the_history_accumulates(slots):
    slots.install("T", TERM, {"type": "sl_episode_a", "scale": 2.0})
    slots.episode = 4
    first_hash = slots.get("T").resolved.spec_hash
    record = slots.swap("T", {"type": "sl_episode_b", "scale": 3.0}, boundary=Boundary.EPISODE)
    assert isinstance(record, SwapRecord) and record.episode == 4
    assert record.slot == "T" and record.kind_id == TERM
    assert record.from_spec_hash == first_hash
    assert record.to_spec_hash == ComponentSpec.parse(TERM, {"type": "sl_episode_b", "scale": 3.0}).hash()
    assert record.at > 0 and set(record.as_dict()) == {
        "slot", "kind_id", "from_name", "to_name", "from_spec_hash", "to_spec_hash",
        "boundary", "episode", "at"}
    slots.swap("T", "sl_episode_a", boundary=Boundary.EPISODE)
    assert [r.to_name for r in slots.history] == ["sl_episode_b", "sl_episode_a"]


def test_swap_accepts_an_already_parsed_spec(slots):
    slots.install("T", TERM, "sl_episode_a")
    spec = ComponentSpec.parse(TERM, {"type": "sl_episode_b", "scale": 5.0})
    slots.swap("T", spec, boundary=Boundary.EPISODE)
    assert slots.get("T").instance.scale == 5.0


# -- the queue ---------------------------------------------------------------------
def test_request_validates_eagerly_and_queues(slots):
    slots.install("T", TERM, "sl_episode_a")
    with pytest.raises(ComponentMissing):
        slots.request("T", "sl_missing")
    with pytest.raises(ComponentParamError):
        slots.request("T", {"type": "sl_episode_b", "bogus": 1})
    assert slots.pending() == {}
    slots.request("T", {"type": "sl_episode_b", "scale": 2.0})
    assert slots.pending() == {"T": {"type": "sl_episode_b", "params": {"scale": 2.0}}}
    assert slots.get("T").component.name == "sl_episode_a"  # nothing happens until a boundary
    slots.request("T", "sl_episode_a")  # a second request replaces the first
    assert slots.pending() == {"T": {"type": "sl_episode_a", "params": {}}}
    assert slots.drop_pending("T") is True and slots.drop_pending("T") is False
    assert slots.pending() == {}


def test_apply_pending_applies_what_the_boundary_permits_and_keeps_the_rest(slots):
    slots.install("T", TERM, "sl_episode_a")
    slots.install("P", POLICY, "sl_static_a")
    slots.request("T", "sl_episode_b")
    slots.request("P", "sl_static_b")
    applied = slots.apply_pending(Boundary.EPISODE)
    assert [r.slot for r in applied] == ["T"]
    assert slots.get("T").component.name == "sl_episode_b"
    assert slots.get("P").component.name == "sl_static_a"
    assert set(slots.pending()) == {"P"}  # a static slot waits for run_start
    applied = slots.apply_pending(Boundary.RUN_START)
    assert [r.slot for r in applied] == ["P"] and slots.pending() == {}


def test_apply_pending_runs_in_slot_order_and_calls_the_migration(slots):
    for path in ("B", "A", "C"):
        slots.install(path, TERM, "sl_episode_a")
    calls = []
    for path in ("B", "A", "C"):
        slots.request(path, "sl_episode_b", migrate=lambda o, n, p=path: calls.append(p))
    assert [r.slot for r in slots.apply_pending(Boundary.EPISODE)] == ["A", "B", "C"]
    assert calls == ["A", "B", "C"]


def test_apply_pending_with_a_failing_build_raises_and_keeps_it_queued(slots):
    slots.install("T", TERM, "sl_episode_a")
    slots.request("T", "sl_episode_b")
    HUB.registry(TERM).unregister("sl_episode_b", scope="run")  # it vanished between request and boundary
    with pytest.raises(ComponentMissing):
        slots.apply_pending(Boundary.EPISODE)
    assert "T" in slots.pending() and slots.get("T").component.name == "sl_episode_a"


def test_fingerprints_report_every_slot_sorted(slots):
    slots.install("B", POLICY, "sl_static_a")
    slots.install("A", TERM, {"type": "sl_episode_a", "scale": 2.0})
    fingerprints = slots.fingerprints()
    assert list(fingerprints) == ["A", "B"]
    assert fingerprints["A"]["name"] == "sl_episode_a" and fingerprints["A"]["origin"] == "project"
    assert fingerprints["A"]["spec_hash"] == ComponentSpec.parse(TERM, {"type": "sl_episode_a", "scale": 2.0}).hash()
    assert len(fingerprints["A"]["source_digest"]) == 16


def test_an_untrusted_table_refuses_an_import_spec(slots):
    from mactf.exceptions import ComponentImportDenied

    untrusted = SlotTable(ComponentResolver(), trusted=False)
    with pytest.raises(ComponentImportDenied):
        untrusted.install("T", TERM, {"import": "acme.mod:Cls"})
    slots.install("T", TERM, "sl_episode_a")
    with pytest.raises(ComponentImportDenied):
        untrusted.install("T2", TERM, {"import": "acme.mod:Cls"})
