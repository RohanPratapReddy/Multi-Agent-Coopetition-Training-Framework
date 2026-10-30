"""Training policies (document 17 section 7): the protocol, the resolver, the default step and 7.10."""

from __future__ import annotations

import copy
from typing import Any, ClassVar, Mapping

import pytest
import torch

from mactf.components import ComponentKind, HUB, register_policy
from mactf.components.policy_resolver import PolicyChoice, resolve_policy
from mactf.components.guards.step import GuardedPolicy
from mactf.exceptions import (
    ComponentLocked,
    ContractError,
    ImproperlyConfigured,
    UnsafePolicyDenied,
)
from mactf.rl.functions.loss import CallableLoss, CompositeLoss
from mactf.rl.loss import LossBreakdown
from mactf.training.policy import (
    DefaultTrainPolicy,
    PolicyHost,
    TrainerPolicy,
    VOID_ADVANTAGE_FLOOR,
    policy_flag,
    policy_name,
)
from mactf.training.protocol import StepResult

from p2_support import configure, make_trainer, snapshot_parameters, tune


def _batch(**kwargs):
    generator = torch.Generator().manual_seed(2)
    base = {"states": torch.randn(8, 4, generator=generator), "targets": torch.zeros(8, dtype=torch.long),
            "signal": 0.4, "void": False, "floor": 0.0}
    base.update(kwargs)
    return base


# -- the default policy is the old step -------------------------------------------------------------
def test_the_default_policy_is_the_old_step(settings, tmp_path):
    """Compared against the recorded 19.4 fixtures (the exact scenario of ``default_identity``)."""
    import default_identity as recorder

    fixtures = recorder.load_fixtures()
    import torch as _torch

    if fixtures["run"]["torch"] != _torch.__version__.split("+")[0].rsplit(".", 1)[0]:
        pytest.skip("fixtures were recorded with another torch")
    replay = recorder.record_all()
    assert recorder.diff(replay, fixtures) == []


def test_a_default_trainer_has_the_default_policy_and_no_configured_one(settings, tmp_path):
    tune(settings)
    trainer = make_trainer(settings, tmp_path).build()
    assert isinstance(trainer._policy, DefaultTrainPolicy)
    assert trainer._trainer_policy is None and trainer._policy_origin.value == "builtin"
    assert trainer._policy_guard is None and trainer.distill is None
    slot = trainer.slots.get("TRAINER.POLICY")
    assert slot.component.name == "default" and slot.component.origin.value == "builtin"
    assert trainer.slots.fingerprints()["TRAINER.POLICY"]["name"] == "default"


def test_the_default_policy_describes_itself():
    policy = DefaultTrainPolicy(advantage_masks="void")
    assert policy.describe() == {"name": "default", "advantage_masks": "void", "hooks": []}
    assert policy.consumes_functions == {"loss", "advantage", "action", "regularization", "metric"}
    assert VOID_ADVANTAGE_FLOOR == -1.0


def test_train_step_before_attach_is_a_contract_error():
    with pytest.raises(ContractError, match="before attach"):
        DefaultTrainPolicy().train_step(object(), {})


def test_constructor_hooks_beat_the_function_book(settings, tmp_path, component_overlay):
    from p2_support import ClippedAdvantage
    from mactf.components import register_function

    register_function("advantage", "clipped", ClippedAdvantage, scope="run")
    tune(settings)
    configure(settings, RL={"ADVANTAGE": {"type": "clipped", "params": {"clip": 1.0}}})
    trainer = make_trainer(settings, tmp_path).build()
    mine = ClippedAdvantage(clip=2.0)
    policy = DefaultTrainPolicy(advantage=mine)
    policy.attach(trainer)
    assert policy._hook("advantage", "peer-1") is mine
    assert DefaultTrainPolicy()._hook("advantage", "peer-1") is None            # unattached: nothing


def test_the_action_and_metric_hooks_are_read(settings, tmp_path, component_overlay):
    from mactf.components import register_function

    chosen = []

    class AlwaysZero:
        def __call__(self, logits, *, sample):
            chosen.append(sample)
            return torch.zeros(logits.shape[0], dtype=torch.long)

    class CountMetric:
        def __call__(self, batch, output):
            return {"rows": float(batch["actions"].numel())}

    register_function("action", "always_zero", AlwaysZero, scope="run")
    register_function("metric", "count", CountMetric, scope="run")
    tune(settings)
    configure(settings, RL={"ACTION": {"type": "always_zero"}, "METRIC": {"type": "count"}})
    trainer = make_trainer(settings, tmp_path).build()
    chosen.clear()                                    # the contract probe called it at build
    result = trainer.train_step(trainer.tree.get("peer-1"), _batch())
    assert chosen == [False]
    assert result.metrics["metric.rows"] == 8.0
    assert "lr" in result.metrics


# -- precedence and conflicts (7.5) -----------------------------------------------------------------------
class Mine:
    name: ClassVar[str] = "mine"
    unsafe: ClassVar[bool] = False
    self_stepping: ClassVar[bool | None] = False
    requires_terms: ClassVar[frozenset[str]] = frozenset()
    composes_distillation: ClassVar[bool] = False

    def __init__(self, *, rate: float = 0.1) -> None:
        self.rate = rate
        self.host: Any = None
        self.attached = 0
        self.setups: list[str] = []
        self.epoch_ends = 0

    def attach(self, host: Any) -> None:
        self.host = host
        self.attached += 1

    def setup(self, node: Any) -> None:
        self.setups.append(node.path)

    def on_epoch_end(self, tree, result) -> None:
        self.epoch_ends += 1

    def card_metadata(self, path: str) -> dict[str, Any]:
        return {"rate": self.rate}

    def state_dict(self) -> dict[str, Any]:
        return {"epoch_ends": self.epoch_ends}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self.epoch_ends = int(payload["epoch_ends"])

    def train_step(self, node, batch):
        _, peer = self.host.resolve_node(node)
        logits, _ = peer.adapter(batch["states"])
        loss = logits.pow(2).mean() * self.rate
        return StepResult(loss=loss, breakdown=LossBreakdown(total=float(loss.detach())), stepped=True)


def test_a_spec_policy_is_resolved_wrapped_attached_and_driven(settings, tmp_path, component_overlay):
    register_policy("mine", Mine, scope="run")
    tune(settings)
    configure(settings, TRAINER={"POLICY": {"type": "mine", "params": {"rate": 0.3}}})
    trainer = make_trainer(settings, tmp_path).build()
    inner = trainer._trainer_policy
    assert isinstance(inner, Mine) and inner.rate == 0.3 and inner.attached == 1
    assert isinstance(trainer._policy, GuardedPolicy) and trainer._policy.inner is inner
    assert sorted(inner.setups) == sorted(trainer.node_paths)
    assert trainer._policy_origin.value == "project"
    trainer.run_epoch(0)
    assert inner.epoch_ends == 1
    for card in trainer._epoch_cards[-1].peers.values():
        assert card.metadata["policy"]["rate"] == 0.3
        assert card.metadata["policy"]["step_guard"]["mode"] == "quarantine"
    state = trainer._trainer_state()
    assert state["policy"]["inner"] == {"epoch_ends": 1}
    # a policy's state is written when there is some, and restored
    payload = trainer.state_dict()
    other = make_trainer(settings, tmp_path / "other").build()
    other.load_state_dict(payload, strict=False)
    assert other._trainer_policy.epoch_ends == 1


def test_an_in_code_instance_wins_and_conflicts_with_a_spec(settings, tmp_path, component_overlay):
    register_policy("mine", Mine, scope="run")
    tune(settings)
    instance = Mine()
    trainer = make_trainer(settings, tmp_path, trainer=instance).build()
    assert trainer._trainer_policy is instance and trainer._policy.inner is instance
    assert trainer._policy_origin.value == "runtime"
    configure(settings, TRAINER={"POLICY": "mine"})
    with pytest.raises(ImproperlyConfigured, match="both given"):
        make_trainer(settings, tmp_path / "x", trainer=Mine()).build()


def test_the_policy_slot_is_static(settings, tmp_path, component_overlay):
    register_policy("mine", Mine, scope="run")
    tune(settings)
    trainer = make_trainer(settings, tmp_path).build()
    from mactf.components.kinds import Boundary

    with pytest.raises(ComponentLocked, match="may only change at run_start"):
        trainer.slots.swap("TRAINER.POLICY", "mine", boundary=Boundary.EPISODE)


def test_an_unknown_policy_name_is_a_clear_error_at_build(settings, tmp_path):
    from mactf.exceptions import ComponentMissing

    tune(settings)
    configure(settings, TRAINER={"POLICY": "nope"})
    # The settings load would refuse an unknown name; build reaches the registry with the same facts
    with pytest.raises(ComponentMissing, match="did you mean|nope"):
        make_trainer(settings, tmp_path).build()


def test_the_builtin_policies_are_registered():
    names = HUB.registry("trainer_policy").names()
    assert {"default", "co_distill"} <= set(names)
    default = HUB.registry("trainer_policy").get("default")
    assert default.origin.value == "builtin" and default.mutability.value == "static"


def test_policy_helpers():
    assert policy_flag(object(), "unsafe", False) is False
    assert policy_name(Mine()) == "mine"
    assert policy_name(lambda: None) == "<lambda>"
    # ``TrainerPolicy`` is documentation: only ``train_step`` is required, the rest is read with getattr
    assert callable(Mine().train_step) and TrainerPolicy.__name__ == "TrainerPolicy"


def test_the_trainer_implements_the_host_surface(settings, tmp_path):
    tune(settings)
    trainer = make_trainer(settings, tmp_path).build()
    assert isinstance(trainer, PolicyHost)
    node, peer = trainer.resolve_node(trainer.tree.get("peer-1"))
    assert node.path == peer.name == "peer-1"
    assert trainer.loss_for("peer-1") is trainer.loss_function


# -- distillation (7.5, 7.9) ----------------------------------------------------------------------------
VIEW = {"kind": "fixed_random", "out_dim": 8}


def _teacher(name: str = "t1") -> dict:
    return {"name": name, "kind": "module", "factory": "torch.nn.Linear",
            "kwargs": {"in_features": 4, "out_features": 3}, "view": dict(VIEW),
            "trust": {"trust_init": 1.0}}


def _distillation(**overrides) -> dict:
    value = {"TEACHERS": [_teacher()], "OUT_DIM": 8, "TAU": 0.5, "VIEWS": {"*": dict(VIEW)}}
    value.update(overrides)
    return value


def test_distillation_alone_runs_the_step_itself_unwrapped(settings, tmp_path):
    from mactf.distill import CoDistillStep

    tune(settings)
    trainer = make_trainer(settings, tmp_path, distillation=_distillation()).build()
    assert isinstance(trainer._policy, CoDistillStep) and trainer._trainer_policy is trainer._policy
    assert trainer.distill is trainer._policy and trainer._policy_guard is None
    assert trainer.slots.get("TRAINER.POLICY").component.name == "co_distill"
    assert trainer.slots.get("TRAINER.POLICY").component.origin.value == "builtin"


def test_a_policy_that_does_not_compose_distillation_conflicts_with_it(settings, tmp_path, component_overlay):
    register_policy("mine", Mine, scope="run")
    tune(settings)
    configure(settings, TRAINER={"POLICY": "mine"})
    trainer = make_trainer(settings, tmp_path, distillation=_distillation())
    with pytest.raises(ImproperlyConfigured, match="composes_distillation"):
        trainer.build()


def test_a_policy_that_composes_distillation_finds_the_step_at_host_distill(settings, tmp_path, component_overlay):
    from mactf.distill import CoDistillStep

    class Composer(Mine):
        name = "composer"
        composes_distillation = True

        def attach(self, host):
            super().attach(host)
            self.step = host.distill

    register_policy("composer", Composer, scope="run")
    tune(settings)
    configure(settings, TRAINER={"POLICY": "composer"})
    trainer = make_trainer(settings, tmp_path, distillation=_distillation()).build()
    assert isinstance(trainer._trainer_policy.step, CoDistillStep) and trainer.distill is trainer._trainer_policy.step
    # an in-code composing instance is accepted by the constructor too
    other = make_trainer(settings, tmp_path / "x", distillation=_distillation(), trainer=None)
    assert other.distillation.enabled


def test_a_trainer_instance_that_composes_distillation_is_accepted(settings, tmp_path):
    class Composer(Mine):
        composes_distillation = True

    tune(settings)
    trainer = make_trainer(settings, tmp_path, trainer=Composer(), distillation=_distillation()).build()
    assert trainer.distill is not None and isinstance(trainer._trainer_policy, Composer)


def test_co_distill_by_name_needs_distillation_on(settings, tmp_path):
    tune(settings)
    configure(settings, TRAINER={"POLICY": "co_distill"})
    with pytest.raises(ImproperlyConfigured, match="DISTILLATION.ENABLED is false"):
        make_trainer(settings, tmp_path).build()


def test_co_distill_by_name_delegates_to_the_one_step(settings, tmp_path):
    from mactf.distill import CoDistillStep

    tune(settings)
    configure(settings, TRAINER={"POLICY": "co_distill"})
    trainer = make_trainer(settings, tmp_path, distillation=_distillation()).build()
    delegate = trainer._policy
    assert type(delegate).__name__ == "CoDistillPolicy" and isinstance(delegate._step, CoDistillStep)
    assert delegate._step is trainer.distill
    trainer.run_epoch(0)
    assert trainer._trainer_state()["distillation"]                         # still under its own key
    assert "policy" not in trainer._trainer_state()                         # and not duplicated
    for card in trainer._epoch_cards[-1].peers.values():
        assert "distillation" in card.metadata


# -- unsafe: the resolver alone -----------------------------------------------------------------------------
def test_resolve_policy_refuses_unsafe_without_the_flag_and_names_the_fix():
    class Host:
        distill = None

    class Unsafe(Mine):
        unsafe = True

    class Slots:
        def adopt(self, *args, **kwargs):
            from mactf.components.slots import Slot

            return type("S", (), {"resolved": type("R", (), {"fingerprint": lambda self: {}})()})()

    with pytest.raises(UnsafePolicyDenied) as raised:
        resolve_policy(code_instance=Unsafe(), settings={"TRAINER": {}}, distillation=None,
                       slots=Slots(), host=Host(), distill_factory=lambda: None)
    assert raised.value.code == "E009" and "TRAINER.ALLOW_UNSAFE_POLICY" in str(raised.value)


# -- 7.10: the worked example ---------------------------------------------------------------------------------------
class AnchoredPolicy:
    """A policy that runs the default-style step plus an anchor to the reference (7.10)."""

    name: ClassVar[str] = "anchored"
    unsafe: ClassVar[bool] = False
    self_stepping: ClassVar[bool | None] = False
    requires_terms: ClassVar[frozenset[str]] = frozenset({"compliance_floor"})
    composes_distillation: ClassVar[bool] = False

    def __init__(self, *, anchor_coefficient: float = 0.05) -> None:
        if anchor_coefficient < 0:
            raise ValueError("anchor_coefficient must be >= 0")
        self.anchor_coefficient = float(anchor_coefficient)
        self._host = None

    def attach(self, host):
        self._host = host

    def train_step(self, node, batch):
        host = self._host
        assert host is not None
        _, peer = host.resolve_node(node)
        states = batch["states"]
        logits, values = peer.adapter(states)
        with torch.no_grad():
            ref_logits, _ = host.reference.adapter(states)
        anchor = self.anchor_coefficient * torch.nn.functional.mse_loss(logits, ref_logits)
        floor = float(batch.get("floor", 0.0))
        compliance = torch.tensor(max(-floor, 0.0), device=logits.device)
        signal = float(batch.get("signal", 0.0))
        task = -signal * torch.log_softmax(logits, dim=-1).mean()
        total = task + anchor + compliance
        breakdown = LossBreakdown(policy=float(task.detach()), compliance=float(compliance.detach()),
                                  total=float(total.detach()), extras={"anchor": float(anchor.detach())})
        return StepResult(loss=total, breakdown=breakdown, stepped=True)


def test_the_worked_example_policy_runs_under_the_guard(settings, tmp_path, component_overlay):
    register_policy("anchored", AnchoredPolicy, version="1", scope="run")
    tune(settings)
    configure(settings, TRAINER={"POLICY": {"type": "anchored", "params": {"anchor_coefficient": 0.1}},
                                 "STEP_GUARD": {"MODE": "quarantine", "MAX_CONSECUTIVE": 5}})
    trainer = make_trainer(settings, tmp_path).build()
    before = snapshot_parameters(trainer)
    result = trainer.fit(epochs=2)
    after = snapshot_parameters(trainer)
    assert any(not torch.equal(before[k], after[k]) for k in before)               # it trained
    assert trainer._policy_guard.checked == 4 and trainer._policy_guard.quarantined == 0
    assert trainer._policy_guard.config.max_consecutive == 5
    # a breached floor with a policy that does charge it is fine; the guard saw the evidence
    r = trainer.train_step(trainer.tree.get("peer-1"), _batch(floor=-0.5))
    assert r.breakdown.compliance == pytest.approx(0.5) and r.metrics.get("quarantined") is None
    assert trainer._policy.card_metadata("peer-1")["step_guard"]["checked"] == 5
    del result


def test_the_worked_example_declared_term_is_enforced_when_it_is_skipped(settings, tmp_path, component_overlay):
    class Forgetful(AnchoredPolicy):
        name = "forgetful"

        def train_step(self, node, batch):
            result = super().train_step(node, batch)
            result.breakdown.compliance = 0.0
            return result

    register_policy("forgetful", Forgetful, scope="run")
    tune(settings)
    configure(settings, TRAINER={"POLICY": "forgetful"})
    trainer = make_trainer(settings, tmp_path).build()
    result = trainer.train_step(trainer.tree.get("peer-1"), _batch(floor=-0.5))
    assert result.metrics.get("quarantined") == 1.0
    assert trainer._policy_guard.events[-1].violations[0].code == "S004"


# -- the loss the trainer resolves ------------------------------------------------------------------------------------
def test_an_in_code_custom_loss_is_extended_with_the_compliance_term(settings, tmp_path):
    tune(settings)
    trainer = make_trainer(settings, tmp_path, loss=lambda batch: batch["values"].mean()).build()
    loss = trainer.loss_function
    assert type(loss).__name__ == "WeightedSumLoss" and isinstance(loss.terms[0], CallableLoss)
    assert loss.weights == [1.0, 1.0] and "compliance_floor" in loss.required_fields


def test_the_default_loss_object_is_untouched(settings, tmp_path):
    tune(settings)
    assert type(make_trainer(settings, tmp_path).build().loss_function) is CompositeLoss


def test_loss_compliance_mode_off_and_enforce_for_an_in_code_loss(settings, tmp_path):
    tune(settings)
    configure(settings, TRAINER={"LOSS_COMPLIANCE": "off"})
    off = make_trainer(settings, tmp_path, loss=lambda batch: batch["values"].mean()).build()
    assert isinstance(off.loss_function, CallableLoss)
    configure(settings, TRAINER={"LOSS_COMPLIANCE": "enforce"})
    with pytest.raises(Exception, match="no compliance_floor term"):
        make_trainer(settings, tmp_path / "e", loss=lambda batch: batch["values"].mean()).build()


# -- the legacy forms, once more through the whole trainer ---------------------------------------------------------------------
def test_a_policy_object_with_only_train_step_still_works(settings, tmp_path):
    class Bare:
        def train_step(self, node, batch):
            return StepResult(loss=torch.tensor(0.25), breakdown=LossBreakdown(total=0.25), stepped=True)

    tune(settings)
    trainer = make_trainer(settings, tmp_path, trainer=Bare()).build()
    result = trainer.run_epoch(0)
    assert result.mean_loss == pytest.approx(0.25)
    assert trainer._policy_guard.config.mode == "warn"


def test_a_trainer_instance_in_code_warns_first(settings, tmp_path, caplog):
    """19.3.3 row 28 / risk R5: an instance that predates the guard is warned about, then runs."""
    import logging

    class Nan(Mine):
        def train_step(self, node, batch):
            _, peer = self.host.resolve_node(node)
            logits, _ = peer.adapter(batch["states"])
            return StepResult(loss=logits.mean() * float("nan"), breakdown=LossBreakdown(), stepped=True)

    tune(settings)
    trainer = make_trainer(settings, tmp_path, trainer=Nan()).build()
    with caplog.at_level(logging.WARNING, logger="mactf.components.step_guard"):
        result = trainer.train_step(trainer.tree.get("peer-1"), _batch())
    assert "step guard warned" in caplog.text and "S001" in caplog.text
    assert result.metrics.get("quarantined") is None       # it ran, with the nan the guard reported
    assert trainer._policy_guard.config.mode == "warn"
