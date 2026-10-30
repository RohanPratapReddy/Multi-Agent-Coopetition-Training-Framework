"""``FunctionBook`` (document 17 section 8.4-8.8) and the RL invariants R1-R12 of section 8.10."""

from __future__ import annotations

import logging

import pytest
import torch

from mactf.components import register_function
from mactf.components.function_book import FunctionBook
from mactf.components.kinds import Boundary, Origin
from mactf.components.migrations import (
    FAMILY_DEFAULTS,
    carry_parameters,
    carry_state,
    fail_if_stateful,
    is_stateful,
    migration_for,
    reset,
    state_digest,
)
from mactf.exceptions import (
    ComponentContractError,
    ComponentLocked,
    ComponentMigrationError,
    ComponentSpecError,
    ImproperlyConfigured,
)
from mactf.rl.functions.loss import ChainedLoss, CompositeLoss, PolicyGradientLoss, WeightedSumLoss

from p2_support import (
    ClippedAdvantage,
    HuberValueLoss,
    L1AnnealRegularization,
    MedianSynthesis,
    configure,
    hierarchy_trainer,
    make_trainer,
    tune,
)


@pytest.fixture
def examples(component_overlay):
    register_function("advantage", "clipped", ClippedAdvantage, scope="run")
    register_function("regularization", "l1_anneal", L1AnnealRegularization, scope="run")
    register_function("loss", "huber_value", HuberValueLoss, scope="run")
    register_function("synthesis", "median", MedianSynthesis, scope="run")
    return component_overlay


def _built(settings, tmp_path, rl=None, **kwargs):
    tune(settings)
    if rl is not None:
        configure(settings, RL=rl)
    return make_trainer(settings, tmp_path, **kwargs).build()


# -- R1: nothing configured, nothing changes ---------------------------------------------------
def test_default_run_unchanged_every_consumer_falls_back(settings, tmp_path):
    from mactf.components.kinds import RL_FAMILIES

    trainer = _built(settings, tmp_path)
    assert all(trainer.functions.get(family, "peer-1") is None for family in RL_FAMILIES)
    assert trainer._function_warnings == []
    assert trainer.loss_for("peer-1") is trainer.loss_function
    assert isinstance(trainer.loss_function, CompositeLoss)         # no wrapper: I1
    assert trainer.functions.state_dict() == {"version": 1, "entries": []}
    assert "functions" not in trainer._trainer_state()


# -- R2: every configured family has a consumer -------------------------------------------------
def test_unconsumed_family_is_error(settings, tmp_path, examples):
    with pytest.raises(ImproperlyConfigured, match=r"E041.*RL\.GRADIENT.*nothing consumes it"):
        _built(settings, tmp_path, rl={"GRADIENT": {"type": "clip"}})


def test_unconsumed_family_is_a_warning_when_asked(settings, tmp_path, examples, caplog):
    with caplog.at_level(logging.WARNING, logger="mactf.components"):
        trainer = _built(settings, tmp_path, rl={"GRADIENT": {"type": "clip"}, "UNCONSUMED": "warn"})
    assert any(w.startswith("W010") for w in trainer._function_warnings)
    assert "W010" in caplog.text


def test_a_policy_that_lists_a_family_consumes_it(settings, tmp_path, examples):
    from mactf.components import register_policy

    class UsesGradient:
        name = "uses_gradient"
        consumes_functions = frozenset({"loss", "gradient"})
        self_stepping = True

        def train_step(self, node, batch):  # pragma: no cover - never run
            raise NotImplementedError

    register_policy("uses_gradient", UsesGradient, scope="run")
    configure(settings, TRAINER={"POLICY": "uses_gradient"})
    trainer = _built(settings, tmp_path, rl={"GRADIENT": {"type": "clip"}})
    assert trainer.functions.get("gradient", "peer-1").describe()["type"] == "clip"


def test_synthesis_and_rollup_need_a_hierarchy_e045(settings, tmp_path, examples):
    with pytest.raises(ImproperlyConfigured, match="E045"):
        _built(settings, tmp_path, rl={"SYNTHESIS": {"type": "median"}})


# -- E040: one source of truth per default ------------------------------------------------------
def test_e040_loss_block_and_rl_loss(settings, tmp_path, examples):
    configure(settings, LOSS={"TYPE": "composite", "KL_COEF": 0.02})
    with pytest.raises(ImproperlyConfigured, match="E040"):
        _built(settings, tmp_path, rl={"LOSS": {"type": "huber_value"}})


def test_e040_loss_argument_and_rl_loss(settings, tmp_path, examples):
    with pytest.raises(ImproperlyConfigured, match="E040"):
        _built(settings, tmp_path, rl={"LOSS": {"type": "huber_value"}}, loss=CompositeLoss())


def test_e040_synthesis_argument_and_rl_synthesis(settings, tmp_path, examples):
    with pytest.raises(ImproperlyConfigured, match=r"E040.*synthesis"):
        tune(settings)
        configure(settings, RL={"SYNTHESIS": {"type": "median"}})
        hierarchy_trainer(settings, tmp_path, synthesis="consensus").build()


def test_an_override_may_sit_on_top_of_an_in_code_default(settings, tmp_path, examples):
    tune(settings)
    configure(settings, RL={"OVERRIDES": {"SYNTHESIS": {"peer-1": {"type": "median"}}}})
    trainer = hierarchy_trainer(settings, tmp_path, synthesis="consensus").build()
    assert type(trainer.synthesis_for("peer-1")).__name__ == "TensorSynthesisAdapter"
    assert type(trainer.synthesis_for("peer-1/sub-1")).__name__ == "ConsensusSynthesis"


# -- R7: deterministic resolution of overrides -----------------------------------------------------
def test_override_specificity_exact_path_beats_a_glob(settings, tmp_path, examples):
    trainer = _built(settings, tmp_path, rl={
        "ADVANTAGE": {"type": "clipped", "params": {"clip": 3.0}},
        "OVERRIDES": {"ADVANTAGE": {
            "peer-*": {"type": "clipped", "params": {"clip": 1.0}},
            "peer-2": {"type": "clipped", "params": {"clip": 0.5}},
        }},
    })
    functions = trainer.functions
    assert functions.get("advantage", "peer-1").clip == 1.0      # the glob beats the default
    assert functions.get("advantage", "peer-2").clip == 0.5      # the exact path beats the glob


def test_override_specificity_the_longer_glob_wins(settings, tmp_path, examples):
    trainer = _built(settings, tmp_path, rl={"OVERRIDES": {"ADVANTAGE": {
        "*": {"type": "clipped", "params": {"clip": 9.0}},
        "peer-*": {"type": "clipped", "params": {"clip": 2.0}},
    }}})
    assert trainer.functions.get("advantage", "peer-1").clip == 2.0


def test_e043_two_equally_specific_globs_are_ambiguous(settings, tmp_path, examples):
    with pytest.raises(ComponentSpecError, match=r"E043.*equally specific"):
        _built(settings, tmp_path, rl={"OVERRIDES": {"ADVANTAGE": {
            "peer-?": {"type": "clipped", "params": {"clip": 1.0}},
            "*eer-1": {"type": "clipped", "params": {"clip": 2.0}},
        }}})


def test_e044_a_glob_that_matches_nothing(settings, tmp_path, examples):
    rl = {"OVERRIDES": {"ADVANTAGE": {"ghost-*": {"type": "clipped"}}}}
    with pytest.raises(ImproperlyConfigured, match=r"E044.*ghost"):
        _built(settings, tmp_path, rl=rl)


def test_w011_the_same_glob_is_a_warning_when_not_strict_or_after_a_mutation(settings, tmp_path, examples):
    trainer = _built(settings, tmp_path, rl={
        "STRICT_GLOBS": False, "OVERRIDES": {"ADVANTAGE": {"ghost-*": {"type": "clipped"}}}})
    assert any(w.startswith("W011") for w in trainer._function_warnings)
    strict = _built(settings, tmp_path / "again", rl={
        "STRICT_GLOBS": True, "OVERRIDES": {"ADVANTAGE": {"peer-1": {"type": "clipped"}}}})
    assert strict.functions.validate(["peer-2"], policy=strict._policy, initial=False)[0].startswith("W011")


def test_e042_a_rollup_has_no_per_node_override(settings, tmp_path, examples):
    trainer = _built(settings, tmp_path)
    with pytest.raises(ImproperlyConfigured, match="E042"):
        trainer.functions.install_override("rollup", "peer-1", {"type": "mean"})
    with pytest.raises(ImproperlyConfigured, match="E042"):
        trainer.functions.swap_override("rollup", "peer-1", {"type": "mean"}, boundary=Boundary.EPISODE)


# -- R8: one instance per (slot, node) ------------------------------------------------------------------
def test_instances_are_per_node(settings, tmp_path, examples):
    trainer = _built(settings, tmp_path, rl={"REGULARIZATION": {"type": "l1_anneal", "params": {"steps": 4}}})
    module = torch.nn.Linear(2, 2)
    first, second = trainer.functions.get("regularization", "peer-1"), trainer.functions.get("regularization", "peer-2")
    assert first is not second
    assert trainer.functions.get("regularization", "peer-1") is first   # cached
    first(module), first(module)
    assert first._calls == 2 and second._calls == 0                    # no state leaked between nodes


def test_an_adopted_instance_is_shared_as_given(settings, tmp_path, examples):
    trainer = _built(settings, tmp_path)
    mine = L1AnnealRegularization(steps=3)
    trainer.functions.adopt_default("regularization", mine, name="mine")
    assert trainer.functions.get("regularization", "peer-1") is mine
    assert trainer.functions.get("regularization", "peer-2") is mine
    assert trainer.functions.fingerprints()["RL.regularization"]["origin"] == "runtime"


# -- R3: a loss always carries the compliance term ---------------------------------------------------------
def test_override_loss_gets_compliance(settings, tmp_path, examples):
    """The exit criterion: neither the default nor an override can leave the charge out."""
    trainer = _built(settings, tmp_path, rl={
        "LOSS": {"type": "huber_value", "params": {"delta": 0.5}},
        "OVERRIDES": {"LOSS": {"peer-2": {"type": "huber_value", "params": {"delta": 2.0}}}},
    })
    generator = torch.Generator().manual_seed(1)
    values = torch.randn(4, 1, generator=generator)
    for path, delta in (("peer-1", 0.5), ("peer-2", 2.0)):
        loss = trainer.loss_for(path)
        assert isinstance(loss, WeightedSumLoss)                       # extended, not rejected
        assert loss.terms[0].delta == delta
        assert "compliance_floor" in loss.required_fields
        batch = {"values": values, "value_targets": torch.zeros(4, 1), "compliance_floor": torch.tensor(-0.75)}
        breached, parts = loss.compute(batch)
        clean, clean_parts = loss.compute({**batch, "compliance_floor": torch.tensor(0.0)})
        assert parts.compliance == pytest.approx(0.75) and clean_parts.compliance == 0.0
        assert float(breached) == pytest.approx(float(clean) + 0.75)


def test_override_loss_gets_compliance_through_a_real_step(settings, tmp_path, examples):
    trainer = _built(settings, tmp_path, rl={"OVERRIDES": {"LOSS": {"peer-2": {"type": "huber_value"}}}})
    trainer.run_epoch(0)
    assert trainer.global_step == 2
    result = trainer.train_step(trainer.tree.get("peer-2"), {
        "states": torch.randn(8, 4), "targets": torch.zeros(8, dtype=torch.long),
        "signal": 0.0, "void": False, "floor": -0.5})
    assert result.breakdown.compliance == pytest.approx(0.5)
    assert any(key.startswith("term:") for key in result.breakdown.extras)


def test_loss_compliance_off_leaves_the_loss_alone(settings, tmp_path, examples):
    configure(settings, TRAINER={"LOSS_COMPLIANCE": "off"})
    trainer = _built(settings, tmp_path, rl={"LOSS": {"type": "huber_value"}})
    assert isinstance(trainer.loss_for("peer-1"), HuberValueLoss)


def test_loss_compliance_enforce_refuses_a_loss_that_omits_the_term(settings, tmp_path, examples):
    configure(settings, TRAINER={"LOSS_COMPLIANCE": "enforce"})
    trainer = _built(settings, tmp_path)
    with pytest.raises(Exception, match="compliance_floor"):
        trainer.functions.install_default("loss", {"type": "huber_value"})
        trainer.functions.get("loss", "peer-1")


# -- R9: swaps are atomic ---------------------------------------------------------------------------------------
def _swap_setup(settings, tmp_path, examples):
    trainer = _built(settings, tmp_path, rl={"REGULARIZATION": {"type": "l1_anneal", "params": {"steps": 8}}})
    module = torch.nn.Linear(2, 2)
    old = trainer.functions.get("regularization", "peer-1")
    old(module), old(module), old(module)
    return trainer, old, module


def test_swap_is_atomic_when_the_probe_fails(settings, tmp_path, examples):
    class Broken(L1AnnealRegularization):
        def __call__(self, module):
            return torch.tensor(float("nan"))

    register_function("regularization", "broken_reg", Broken, scope="run")
    trainer, old, _ = _swap_setup(settings, tmp_path, examples)
    slot_before = trainer.functions._slots.get("RL.regularization").resolved
    with pytest.raises(ComponentContractError) as raised:
        trainer.functions.swap_default("regularization", {"type": "broken_reg"}, boundary=Boundary.EPISODE)
    assert raised.value.code == "E047" and "F004" in str(raised.value)
    assert trainer.functions._slots.get("RL.regularization").resolved is slot_before
    assert trainer.functions.get("regularization", "peer-1") is old and old._calls == 3


def test_swap_is_atomic_when_the_migration_refuses(settings, tmp_path, examples):
    trainer, old, _ = _swap_setup(settings, tmp_path, examples)
    history = len(trainer.slots.history)
    with pytest.raises(ComponentMigrationError, match="migrate: fail"):
        trainer.functions.swap_default("regularization", {"type": "l1_anneal", "params": {"steps": 99}},
                                       boundary=Boundary.EPISODE, migrate="fail")
    assert trainer.functions.get("regularization", "peer-1") is old
    assert old.steps == 8 and old._calls == 3
    assert len(trainer.slots.history) == history


def test_swap_is_atomic_when_the_boundary_is_too_early(settings, tmp_path, examples):
    from mactf.components.kinds import Mutability

    class Frozen(L1AnnealRegularization):
        pass

    register_function("regularization", "static_reg", Frozen, scope="run", mutability=Mutability.STATIC)
    trainer, old, _ = _swap_setup(settings, tmp_path, examples)
    with pytest.raises(ComponentLocked):
        trainer.functions.swap_default("regularization", {"type": "static_reg"}, boundary=Boundary.EPISODE,
                                       migrate="reset")
    assert trainer.functions.get("regularization", "peer-1") is old and old._calls == 3


def test_swap_is_atomic_when_the_slot_is_locked(settings, tmp_path, examples):
    from mactf.components import HUB, HubPolicy

    trainer, old, _ = _swap_setup(settings, tmp_path, examples)
    with HUB.scoped("lock", policy=HubPolicy(locked=("slot:RL.regularization",))):
        with pytest.raises(ComponentLocked):
            trainer.functions.swap_default("regularization", {"type": "l1_anneal"},
                                           boundary=Boundary.EPISODE, migrate="reset")
        with pytest.raises(ComponentLocked):
            trainer.functions.request("regularization", None, {"type": "l1_anneal"})
    assert trainer.functions.get("regularization", "peer-1") is old


def test_a_successful_swap_replaces_every_live_node_and_carries_the_counter(settings, tmp_path, examples):
    trainer, old, module = _swap_setup(settings, tmp_path, examples)
    other = trainer.functions.get("regularization", "peer-2")
    other(module)
    record = trainer.functions.swap_default(
        "regularization", {"type": "l1_anneal", "params": {"steps": 32}},
        boundary=Boundary.EPISODE, migrate="carry")
    assert record.to_name == "l1_anneal" and record.from_spec_hash != record.to_spec_hash
    new_first, new_second = trainer.functions.get("regularization", "peer-1"), trainer.functions.get("regularization", "peer-2")
    assert new_first is not old and new_first.steps == 32
    assert (new_first._calls, new_second._calls) == (3, 1)           # each node kept its own position


def test_a_queued_swap_is_applied_at_the_next_episode_boundary(settings, tmp_path, examples):
    trainer = _built(settings, tmp_path, rl={"REGULARIZATION": {"type": "l1_anneal", "params": {"steps": 8}}})
    trainer.run_epoch(0)
    before = trainer.functions.get("regularization", "peer-1")
    path = trainer.functions.request("regularization", None, {"type": "l1_anneal", "params": {"steps": 77}},
                                     migrate="carry")
    assert path == "RL.regularization" and trainer.functions.pending()[path]["migrate"] == "carry"
    assert trainer.functions.get("regularization", "peer-1") is before   # not yet
    trainer.run_epoch(1)
    after = trainer.functions.get("regularization", "peer-1")
    assert after is not before and after.steps == 77
    assert after._calls == before._calls + 1                              # carried, then used once
    assert trainer.functions.pending() == {}


def test_a_queued_swap_that_fails_is_reported_not_silent(settings, tmp_path, examples):
    trainer = _built(settings, tmp_path, rl={"REGULARIZATION": {"type": "l1_anneal"}})
    trainer.run_epoch(0)
    trainer.functions.request("regularization", None, {"type": "l1_anneal"}, migrate="fail")
    trainer.run_epoch(1)                                  # the old one holds state, so ``fail`` refuses
    assert trainer.functions.failed_swaps and "migrate: fail" in trainer.functions.failed_swaps[0]["error"]
    assert trainer.functions.pending() == {}


def test_request_validates_and_probes_now(settings, tmp_path, examples):
    trainer = _built(settings, tmp_path, rl={"REGULARIZATION": {"type": "l1_anneal"}})
    with pytest.raises(Exception, match="l1_anneal"):
        trainer.functions.request("regularization", None, {"type": "l1_anneal", "params": {"nonsense": 1}})
    with pytest.raises(ComponentMigrationError, match="unknown migration"):
        trainer.functions.request("regularization", None, {"type": "l1_anneal"}, migrate="sideways")
    assert trainer.functions.pending() == {}
    assert trainer.functions.drop_pending("RL.regularization") is False


# -- removal ------------------------------------------------------------------------------------------------------
def test_removing_a_default_falls_back_to_the_builtin_behaviour(settings, tmp_path, examples):
    trainer = _built(settings, tmp_path, rl={
        "LOSS": {"type": "huber_value"},
        "OVERRIDES": {"LOSS": {"peer-2": {"type": "huber_value", "params": {"delta": 3.0}}}}})
    trainer.functions.remove_default("loss", boundary=Boundary.EPISODE)
    assert trainer.loss_for("peer-1") is trainer.loss_function
    assert trainer.loss_for("peer-2").terms[0].delta == 3.0           # the override survives
    trainer.functions.remove_override("loss", "peer-2", boundary=Boundary.EPISODE)
    assert trainer.loss_for("peer-2") is trainer.loss_function
    assert [record.to_name for record in trainer.slots.history] == ["<removed>", "<removed>"]


def test_removing_a_stateful_function_warns_that_it_discards_state(settings, tmp_path, examples, caplog):
    trainer, old, _ = _swap_setup(settings, tmp_path, examples)
    with caplog.at_level(logging.WARNING, logger="mactf.components"):
        trainer.functions.remove_default("regularization", boundary=Boundary.EPISODE)
    assert "discards the state" in caplog.text
    assert trainer.functions.get("regularization", "peer-1") is None


def test_forget_drops_the_instances_of_a_node_that_left(settings, tmp_path, examples):
    trainer, old, _ = _swap_setup(settings, tmp_path, examples)
    trainer.functions.forget("peer-1")
    assert trainer.functions.get("regularization", "peer-1") is not old


# -- R10: migrations never lose state silently ------------------------------------------------------------------------
def test_migration_reset_logs_and_cleans_the_new_instance(caplog):
    old, new = L1AnnealRegularization(steps=4), L1AnnealRegularization(steps=4)
    module = torch.nn.Linear(2, 2)
    old(module), old(module)
    new._calls = 5                                      # an instance touched by a probe
    with caplog.at_level(logging.WARNING, logger="mactf.components"):
        reset(old, new)
    assert new._calls == 0 and "discards the state" in caplog.text


def test_migration_carry_moves_the_counter():
    old, new = L1AnnealRegularization(steps=4), L1AnnealRegularization(steps=4)
    module = torch.nn.Linear(2, 2)
    first = float(old(module))
    old(module)
    carry_state(old, new)
    assert new._calls == 2 and float(new(module)) < first
    assert state_digest(old) != state_digest(L1AnnealRegularization(steps=4))


def test_migration_carry_from_a_stateful_function_into_a_stateless_one_raises():
    from mactf.rl.functions.support import AccumulateGradient, ClipGradient

    old = AccumulateGradient(n=4)
    old._count = 3
    with pytest.raises(ComponentMigrationError, match="migrate: reset"):
        carry_state(old, ClipGradient())
    new = AccumulateGradient(n=8)
    carry_state(old, new)
    assert new._count == 3                               # the window is neither dropped nor restarted


def test_migration_carry_from_a_stateless_function_is_a_no_op():
    from mactf.rl.functions.support import ClipGradient

    carry_state(ClipGradient(), L1AnnealRegularization())


def test_migration_fail_refuses_state_and_allows_none():
    from mactf.rl.functions.support import ClipGradient

    fail_if_stateful(ClipGradient(), ClipGradient())
    old = L1AnnealRegularization()
    with pytest.raises(ComponentMigrationError, match="migrate: fail"):
        fail_if_stateful(old, L1AnnealRegularization())


def test_migration_carry_parameters_is_all_or_nothing():
    class Learned:
        def __init__(self, width):
            self.layer = torch.nn.Linear(width, 2)

        def parameters(self):
            return list(self.layer.parameters())

    old, same, wider = Learned(3), Learned(3), Learned(5)
    carry_parameters(old, same)
    assert torch.equal(old.layer.weight, same.layer.weight)
    before = wider.layer.weight.detach().clone()
    with pytest.raises(ComponentMigrationError, match="parameter 0"):
        carry_parameters(old, wider)
    assert torch.equal(wider.layer.weight, before)         # nothing was half-copied


def test_migration_selection_order():
    assert FAMILY_DEFAULTS["value"] == "fail" and FAMILY_DEFAULTS["loss"] == "carry"
    assert migration_for(None, family="advantage") is reset                   # no entry: reset
    assert migration_for(None, family="loss") is carry_state
    assert migration_for(None, family="loss", override={"loss": "fail"}) is fail_if_stateful
    assert migration_for("reset", family="loss", override={"loss": "fail"}) is reset   # the request wins
    with pytest.raises(ComponentMigrationError, match="unknown migration"):
        migration_for("sideways")


def test_is_stateful_reads_parameters_and_state_dict_pairs():
    from mactf.rl.functions.support import ClipGradient

    assert not is_stateful(ClipGradient())
    assert is_stateful(L1AnnealRegularization())
    assert is_stateful(torch.nn.Linear(1, 1)) is True


# -- R11: function state in the checkpoint -----------------------------------------------------------------------------------
def test_function_state_roundtrip(settings, tmp_path, examples):
    rl = {"REGULARIZATION": {"type": "l1_anneal", "params": {"steps": 10}}}
    trainer = _built(settings, tmp_path, rl=rl)
    trainer.run_epoch(0)
    saved = trainer.functions.state_dict()
    assert {entry["node"] for entry in saved["entries"]} == {"peer-1", "peer-2"}
    assert trainer._trainer_state()["functions"] == saved
    fresh = _built(settings, tmp_path / "fresh", rl=rl)
    assert fresh.functions.load_state_dict(saved) == []
    assert fresh.functions.get("regularization", "peer-1")._calls == 1


def test_function_state_with_a_changed_spec_is_w012_or_e049(settings, tmp_path, examples):
    trainer = _built(settings, tmp_path, rl={"REGULARIZATION": {"type": "l1_anneal", "params": {"steps": 10}}})
    trainer.run_epoch(0)
    saved = trainer.functions.state_dict()
    changed = _built(settings, tmp_path / "other", rl={"REGULARIZATION": {"type": "l1_anneal", "params": {"steps": 11}}})
    warnings = changed.functions.load_state_dict(saved)
    assert len(warnings) == 2 and all(w.startswith("W012") for w in warnings)
    assert changed.functions.get("regularization", "peer-1")._calls == 0     # nothing was loaded
    with pytest.raises(ComponentMigrationError, match="E049"):
        changed.functions.load_state_dict(saved, strict=True)
    gone = _built(settings, tmp_path / "gone")
    assert all(w.startswith("W012") for w in gone.functions.load_state_dict(saved))


def test_function_state_is_restored_through_the_trainers_own_checkpoint(settings, tmp_path, examples):
    rl = {"REGULARIZATION": {"type": "l1_anneal", "params": {"steps": 10}}}
    first = _built(settings, tmp_path, rl=rl)
    first.run_epoch(0), first.run_epoch(1)
    payload = first.state_dict()
    assert payload["identity"]["trainer"]["functions"]["entries"]
    second = _built(settings, tmp_path / "second", rl=rl)
    second.load_state_dict(payload, strict=False)
    assert second.functions.get("regularization", "peer-1")._calls == first.functions.get("regularization", "peer-1")._calls


# -- R12: the probe runs before the slot is committed --------------------------------------------------------------------------------
def test_probe_runs_before_commit(settings, tmp_path, examples):
    class Nondiff(HuberValueLoss):
        def compute(self, batch):
            loss, parts = super().compute(batch)
            return loss.detach(), parts

    register_function("loss", "nondiff", Nondiff, scope="run")
    with pytest.raises(ComponentContractError) as raised:
        _built(settings, tmp_path, rl={"LOSS": {"type": "nondiff"}})
    assert raised.value.code == "E047" and "F005" in str(raised.value)


def test_rl_verify_off_skips_the_probe_and_all_probes_builtins(settings, tmp_path, examples):
    class Nondiff(HuberValueLoss):
        def compute(self, batch):
            loss, parts = super().compute(batch)
            return loss.detach(), parts

    register_function("loss", "nondiff", Nondiff, scope="run")
    trainer = _built(settings, tmp_path, rl={"VERIFY": "off", "LOSS": {"type": "nondiff"}})
    assert trainer.functions.settings.verify == "off"
    with pytest.raises(ImproperlyConfigured, match="RL.VERIFY"):
        _built(settings, tmp_path / "bad", rl={"VERIFY": "sometimes"})


# -- the description ---------------------------------------------------------------------------------------------------------------
def test_describe_and_fingerprints(settings, tmp_path, examples):
    trainer = _built(settings, tmp_path, rl={
        "ADVANTAGE": {"type": "clipped"}, "OVERRIDES": {"ADVANTAGE": {"peer-2": {"type": "clipped"}}}})
    description = trainer.functions.describe()
    assert set(description["slots"]) == {"RL.advantage", "RL.advantage.peer-2"}
    assert set(trainer.functions.fingerprints()) == set(description["slots"])
    assert trainer.functions.spec("advantage") == {"type": "clipped", "params": {}}
    assert trainer.functions.spec("loss") is None


def test_unknown_rl_keys_are_refused_by_the_book_too(settings, tmp_path, examples):
    trainer = _built(settings, tmp_path)
    with pytest.raises(ImproperlyConfigured, match="unknown RL keys"):
        trainer.functions.configure({"ADVANTAG": {"type": "gae"}})
    with pytest.raises(ImproperlyConfigured, match="RL.OVERRIDES"):
        trainer.functions.configure({"OVERRIDES": {"NOPE": {"x": {"type": "gae"}}}})


def test_w013_is_logged_once_when_a_custom_function_runs_unprobed(settings, tmp_path, component_overlay, caplog):
    """8.10.2: ``RL.VERIFY: off`` costs a ``W013`` at build, once per book."""
    book = FunctionBook.__new__(FunctionBook)
    book.settings = type("S", (), {"verify": "off"})()
    resolved = type("R", (), {"component": type("C", (), {"origin": Origin.PROJECT})()})()
    with caplog.at_level(logging.WARNING, logger="mactf.components"):
        book._verify("loss", resolved)
        book._verify("loss", resolved)
    assert [r.message for r in caplog.records].count("W013: RL.VERIFY is 'off': custom RL functions run without a contract probe") == 1
