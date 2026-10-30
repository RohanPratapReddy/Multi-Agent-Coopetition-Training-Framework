"""The eight worked examples of document 17 section 18 (test 18 of 19.3.3).

Each example is a shipped module under ``mactf/contrib/components/examples``. The test loads it
into a per-test overlay (so the process registry is untouched), merges its settings fragment,
and drives a real two-epoch run through ``TrainerEngine``; the 'what the guard does when it
misbehaves' rows that are cheap and real follow.
"""

from __future__ import annotations

import importlib
import sys

import pytest
import torch
import yaml

from mactf import conf
from mactf.components import HUB, ComponentResolver, ComponentSpec, register
from mactf.contrib.components import examples as shipped
from mactf.contrib.components.examples import EXAMPLES, load_conf, plan_path
from mactf.exceptions import (
    ComplianceGuardViolation,
    ValidationError,
    ComponentBuildError,
    ImproperlyConfigured,
    MutationError,
    UnsafePolicyDenied,
    WeightGuardViolation,
)
from mactf.orchestration.engine import TrainerEngine
from mactf.scoring.base import ScoringKind
from mactf.scoring.pipeline import build_scorer

from examples_support import fragment, load_example, settings_with
from p2_support import configure, hierarchy_trainer, make_trainer, tune
from scoring_helpers import ledger_of, settings_copy

NUMBERS = sorted(EXAMPLES)


def run(trainer, settings):
    """Two real epochs through the engine; returns the engine's summary."""
    summary = TrainerEngine(settings, trainer=trainer).run()
    assert summary["status"] == "completed", summary
    assert summary["epochs_completed"] == 2
    return summary


def trainer_for(number, settings, tmp_path, *, hierarchy=False, peers=3, **extra):
    tune(settings, EPOCHS=2, CHECKPOINT_EVERY=1)
    settings["DEVICE"] = "cpu"
    settings_with(number, settings, **extra)
    conf.validate_settings(settings)
    if hierarchy:
        return hierarchy_trainer(settings, tmp_path)
    return make_trainer(settings, tmp_path, peers=peers)


@pytest.fixture
def scope(component_overlay):
    return component_overlay


# ---------------------------------------------------------------------------------------- the index
def test_the_index_describes_eight_examples_and_importing_it_registers_nothing():
    assert NUMBERS == [f"18.{n}" for n in range(1, 9)]
    code = (
        "import sys; import mactf.contrib.components.examples as e; "
        "assert 'torch' not in sys.modules; "
        "from mactf.components import HUB; "
        "assert all(HUB.registry('scorer').names().count(n) == 0 for n in ('tempered_reward',)); "
        "print(len(e.EXAMPLES))"
    )
    import subprocess

    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "8", out.stderr


@pytest.mark.parametrize("number", NUMBERS)
def test_every_fragment_parses_and_passes_settings_validation(number):
    blocks = load_conf(number)
    assert isinstance(blocks, dict) and blocks
    for module in blocks.get("COMPONENTS", {}).get("MODULES", ()):
        assert module == EXAMPLES[number]["module"]
        assert importlib.util.find_spec(module) is not None
    settings = settings_copy()
    for name, block in blocks.items():
        settings.setdefault(name, {})
        settings[name].update(block) if isinstance(settings[name], dict) else settings.__setitem__(name, block)
    conf.validate_settings(conf.Settings(settings))


@pytest.mark.parametrize("number", [n for n in NUMBERS if EXAMPLES[n]["module"]])
def test_a_module_registers_only_when_it_is_imported(number, scope):
    entry = EXAMPLES[number]
    kind = entry["kind"]
    registries = [HUB.registry(k) for k in entry.get("kinds", (kind,))]
    assert not any(name in r.names() for r in registries for name in entry["names"])
    load_example(number)
    for name in entry["names"]:
        owner = [r for r in registries if name in r.names()]
        assert len(owner) == 1, name
        assert owner[0].get(name).origin.value in ("project", "runtime", "plugin")


def test_the_real_modules_path_loads_the_module(restore_hub):
    """``COMPONENTS.MODULES`` is the opt-in: it imports the module, which registers."""
    from mactf.components.plugins import ensure_components_loaded, reset_for_tests

    name = EXAMPLES["18.1"]["module"]
    sys.modules.pop(name, None)
    reset_for_tests()
    try:
        assert "tempered_reward" not in HUB.registry("scorer").names()
        ensure_components_loaded({"COMPONENTS": load_conf("18.1")["COMPONENTS"]})
        assert HUB.registry("scorer").get("tempered_reward").origin.value == "project"
    finally:
        sys.modules.pop(name, None)


# ------------------------------------------------------------------------- 18.1 a tempered reward
def test_18_1_runs_and_tempers_the_reward(scope, settings, tmp_path):
    load_example("18.1")
    trainer = trainer_for("18.1", settings, tmp_path)
    run(trainer, settings)
    stage = trainer.scorer.get(ScoringKind.REWARD)
    assert type(stage).__name__ == "TemperedReward" and stage.factor == 0.5
    plain = build_scorer(settings_copy())
    ledger = ledger_of({"R1": 4.0})
    card = trainer.scorer.score_peer(ledger, peer="a", episode=1)
    base = plain.score_peer(ledger_of({"R1": 4.0}), peer="a", episode=1)
    assert card.continuous.reward.r_task == base.continuous.reward.r_task * 0.5
    assert card.continuous.reward.r_task > 0


def test_18_1_misbehaviours(scope):
    module = load_example("18.1")
    register("scorer", "inflated_reward", module.InflatedReward, scope="run")
    with pytest.raises(ValueError, match="factor must be in"):
        module.TemperedReward(factor=4.0)
    resolver = ComponentResolver()
    with pytest.raises(ComponentBuildError, match="E008"):
        resolver.resolve(ComponentSpec.parse("scorer", {"type": "tempered_reward", "params": {"factor": 4.0}}), trusted=True)
    with pytest.raises(ImproperlyConfigured, match="E004.*factr.*accepted"):
        resolver.validate(ComponentSpec.parse("scorer", {"type": "tempered_reward", "params": {"factr": 0.5}}),
                          trusted=True)
    inflating = build_scorer(settings_copy(SCORING={"STAGES": {"reward": {"type": "inflated_reward"}}}))
    with pytest.raises(ComplianceGuardViolation, match="C008"):
        inflating.score_peer(ledger_of({"R1": 4.0}), peer="a", episode=1)
    fallback = settings_copy(SCORING={"STAGES": {"reward": {"type": "inflated_reward"}}},
                             COMPLIANCE={"GUARD_MODE": "fallback"})
    card = build_scorer(fallback).score_peer(ledger_of({"R1": 4.0}), peer="a", episode=1)
    reference = build_scorer(settings_copy()).score_peer(ledger_of({"R1": 4.0}), peer="a", episode=1)
    assert card.continuous.reward.r_task == reference.continuous.reward.r_task     # the reference was substituted


# ---------------------------------------------------------------------------- 18.2 a reward term
def test_18_2_runs_and_the_term_is_in_the_manifest(scope, settings, tmp_path):
    load_example("18.2")
    trainer = trainer_for("18.2", settings, tmp_path)
    run(trainer, settings)
    stage = trainer.scorer.get(ScoringKind.REWARD)
    assert [label for label, _w, _t in stage.terms] == ["severity_scaled", "sqrt_magnitude"]
    manifest = trainer._component_manifest()
    names = {entry["name"] for entry in manifest.get("reward_term", {}).values()}
    assert "sqrt_magnitude" in names, manifest
    nine = trainer.scorer.score_peer(ledger_of({"R10": 9.0}), peer="a", episode=1).continuous.reward.r_collab
    four = trainer.scorer.score_peer(ledger_of({"R10": 4.0}, peer="b"), peer="b", episode=2).continuous.reward.r_collab
    assert four > 0 and nine / four == pytest.approx(1.5)                  # sqrt(9) / sqrt(4): diminishing returns


def test_18_2_misbehaviours(scope):
    module = load_example("18.2")
    resolver = ComponentResolver()
    with pytest.raises(ImproperlyConfigured, match=r"E004|R99"):
        resolver.resolve(ComponentSpec.parse("reward_term", {"type": "sqrt_magnitude", "params": {"rules": ["R99"]}}), trusted=True)
    with pytest.raises(ValidationError, match="scale.*finite and >= 0"):       # the term's own error (P1 notes)
        resolver.resolve(ComponentSpec.parse("reward_term", {"type": "sqrt_magnitude", "params": {"scale": -1}}), trusted=True)
    from mactf.exceptions import ComplianceError
    from mactf.scoring.reward_terms import ComposedRewardScorer

    stage = ComposedRewardScorer(config=settings_copy()["SCORING"], terms=[module.NegativeTerm(rules=["R1"])])
    from mactf.scoring.base import ScoreContext

    ctx = ScoreContext(peer="a", episode=1, ledger=ledger_of({"R1": 2.0}), config=settings_copy()["SCORING"],
                       severity_weights=settings_copy()["SEVERITY"]["WEIGHTS"])
    with pytest.raises(ComplianceError, match="finite and non-negative"):
        stage.score(ctx)
    with pytest.raises(ImproperlyConfigured, match="weight must be finite and >= 0"):
        ComposedRewardScorer(config=settings_copy()["SCORING"], terms=[{"type": "sqrt_magnitude", "weight": -1}])
    with pytest.raises(ImproperlyConfigured, match="at least one term"):
        ComposedRewardScorer(config=settings_copy()["SCORING"], terms=[])


# ------------------------------------------------------------------- 18.3 a per-group weight update
def test_18_3_runs_per_group_with_state(scope, settings, tmp_path):
    load_example("18.3")
    trainer = trainer_for("18.3", settings, tmp_path, hierarchy=True)
    run(trainer, settings)
    plan = trainer.weight_plan
    assert plan.customized()
    described = str(plan.describe([label for label in ("root", "peer-1", "peer-2")]))
    assert "ema_softmax" in described
    assert any("ema_softmax" in note for report in trainer.reports for note in report.adjustments)
    state = trainer.weight_plan.state_dict()
    assert "ema_softmax" in str(state)
    assert trainer.weight_guard.mode == "repair"


def test_18_3_misbehaviours(scope, settings, tmp_path):
    module = load_example("18.3")
    resolver = ComponentResolver()
    with pytest.raises(ImproperlyConfigured, match="E004.*alhpa.*accepted.*alpha"):
        resolver.validate(ComponentSpec.parse("weight_update", {"type": "ema_softmax", "params": {"alhpa": 0.5}}),
                          trusted=True)
    register("weight_update", "sums_to_three", module.SumsToThree, scope="run")
    from p3_support import flat

    repaired = flat(settings, tmp_path / "r", {"WEIGHT_UPDATE": "sums_to_three", "WEIGHT_GUARD": {"PROBE": False}}).build()
    repaired.run_epoch(0)
    assert any("weight_guard repaired weights" in note for note in repaired.reports[0].adjustments)
    strict = flat(settings, tmp_path / "e", {"WEIGHT_UPDATE": "sums_to_three",
                                             "WEIGHT_GUARD": {"PROBE": False, "MODE": "enforce"}}).build()
    with pytest.raises(WeightGuardViolation, match="E063"):
        strict.run_epoch(0)
    with pytest.raises(ImproperlyConfigured, match="E064.*probe"):
        flat(settings, tmp_path / "p", {"WEIGHT_UPDATE": "sums_to_three", "WEIGHT_GUARD": {}}).build()
    live = flat(settings, tmp_path / "l", {"WEIGHT_UPDATE": "ema_softmax"}).build()
    live.run_epoch(0)
    from mactf.exceptions import ComponentMissing

    with pytest.raises(ComponentMissing, match="E001"):
        live.request_weight_update(None, "not_registered")
    assert live.weight_plan.describe()                                        # the old strategy still runs
    live.run_epoch(1)


# ------------------------------------------------------------------------- 18.4 a staggered mix
def test_18_4_runs_and_mixes_every_n_epochs(scope, settings, tmp_path):
    load_example("18.4")
    trainer = trainer_for("18.4", settings, tmp_path, hierarchy=True)
    trainer.build()
    calls = []
    original = trainer.strategy_plan.run
    trainer.strategy_plan.run = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
    run(trainer, settings)
    assert calls                                                              # the plan ran every epoch
    assert "every_n" in str(trainer.strategy_plan.describe(["root", "peer-1", "peer-2"]))
    instance = trainer.strategy_plan.instance_for("root")
    assert type(instance).__name__ == "EveryN" and instance.period == 2
    assert instance.state_dict() == {"epoch": 1}                             # the last epoch it coordinated
    assert type(trainer.strategy_plan.instance_for("peer-2")).__name__ != "EveryN"   # the override: independent


def test_18_4_misbehaviours(scope, settings, tmp_path):
    module = load_example("18.4")
    from mactf.components.strategy_plan import StrategyPlan  # noqa: F401  (the plan the trainer builds)
    from p3_support import hier

    with pytest.raises(ComponentBuildError, match="E008"):
        ComponentResolver().resolve(
            ComponentSpec.parse("peer_strategy", {"type": "every_n", "params": {"period": 0}}), trusted=True)
    register("peer_strategy", "legacy_every_n", module.LegacyEveryN, scope="run")
    trainer = hier(settings, tmp_path, {"PEER_STRATEGY_OVERRIDES": {"peer-1": "legacy_every_n"}})
    with pytest.raises(ImproperlyConfigured, match="E072"):
        trainer.build()


# ------------------------------------------------------------------ 18.5 Lion with a step-down
def test_18_5_runs_with_lion_and_a_scheduler(scope, settings, tmp_path):
    load_example("18.5")
    trainer = trainer_for("18.5", settings, tmp_path)
    run(trainer, settings)
    classes = {path: type(opt.backend.optimizer).__name__ for path, opt in trainer.optimizers.items()}
    assert classes == {"peer-1": "Lion", "peer-2": "AdamW", "peer-3": "Lion"}
    assert trainer.optimizers["peer-1"].scheduler_name == "step_then_flat"
    assert trainer.optimizers["peer-2"].scheduler_name == "step_then_flat"     # facade key inherited
    assert trainer.optimizer_plan.customized()


def test_18_5_misbehaviours(scope, settings, tmp_path):
    load_example("18.5")
    resolver = ComponentResolver()
    with pytest.raises(ComponentBuildError, match="E008.*drop_at must be >= 1"):
        resolver.resolve(ComponentSpec.parse("scheduler", {"type": "step_then_flat", "params": {"drop_at": 0}}),
                         trusted=True, inject={"optimizer": torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)})
    with pytest.raises(ImproperlyConfigured, match="E004"):
        resolver.validate(ComponentSpec.parse("optimizer", {"type": "Lion", "params": {"lrr": 1e-4}}), trusted=True)
    tune(settings, EPOCHS=2)
    configure(settings, OPTIMIZER={**load_conf("18.5")["OPTIMIZER"], "OVERRIDES": {"nowhere-*": {"LR": 1e-4}}})
    with pytest.raises(ImproperlyConfigured, match="E050"):
        make_trainer(settings, tmp_path).build()


# --------------------------------------------------------------------------- 18.6 a Huber loss
def test_18_6_runs_with_the_huber_loss(scope, settings, tmp_path):
    load_example("18.6")
    trainer = trainer_for("18.6", settings, tmp_path)
    run(trainer, settings)
    default = trainer.loss_for("peer-1")
    override = trainer.loss_for("peer-2")
    assert [term.delta for term in default.terms if hasattr(term, "delta")] == [0.5]
    assert [term.delta for term in override.terms if hasattr(term, "delta")] == [2.0]
    assert "compliance_floor" in default.required_fields                    # the framework added the term (R3)


def test_18_6_misbehaviours(scope, settings, tmp_path):
    module = load_example("18.6")
    from mactf.components.function_probes import check_function

    check_function("loss", module.HuberValueLoss(), name="huber_value")
    with pytest.raises(ImproperlyConfigured, match="E047.*F004"):
        check_function("loss", module.NonFiniteHuber(), name="non_finite_huber")
    with pytest.raises(ComponentBuildError, match="E008.*delta must be > 0"):
        ComponentResolver().resolve(
            ComponentSpec.parse("rl_function.loss", {"type": "huber_value", "params": {"delta": 0}}), trusted=True)
    # RL.LOSS and an in-code loss= name the same default: E040
    tune(settings, EPOCHS=2)
    configure(settings, RL={"LOSS": "huber_value"})
    with pytest.raises(ImproperlyConfigured, match="E040"):
        make_trainer(settings, tmp_path / "both", loss=lambda batch: batch["values"].mean()).build()


# ------------------------------------------------------------------------------ 18.7 a policy
def test_18_7_runs_under_the_step_guard(scope, settings, tmp_path):
    load_example("18.7")
    trainer = trainer_for("18.7", settings, tmp_path)
    run(trainer, settings)
    assert type(trainer._policy).__name__ == "GuardedPolicy"
    assert trainer._policy_guard.checked > 0 and trainer._policy_guard.quarantined == 0
    assert trainer._policy_guard.config.max_consecutive == 5
    assert trainer.slots.get("TRAINER.POLICY").component.name == "anchored"


def test_18_7_misbehaviours(scope, settings, tmp_path):
    module = load_example("18.7")
    register("trainer_policy", "forgetful", module.Forgetful, scope="run")
    register("trainer_policy", "greedy", module.Greedy, scope="run")
    from p2_support import make_trainer as make

    tune(settings, EPOCHS=2)
    configure(settings, TRAINER={"POLICY": "forgetful"})
    trainer = make(settings, tmp_path / "f").build()
    result = trainer.train_step(trainer.tree.get("peer-1"), {"states": torch.zeros(4, 4), "floor": -0.5, "signal": 0.0})
    assert result.metrics.get("quarantined") == 1.0
    assert trainer._policy_guard.events[-1].violations[0].code == "S004"
    configure(settings, TRAINER={"POLICY": "greedy"})
    with pytest.raises(UnsafePolicyDenied, match="E009.*ALLOW_UNSAFE_POLICY"):
        make(settings, tmp_path / "g").build()
    # a policy is static: it cannot be swapped on a live run (E003)
    configure(settings, TRAINER={"POLICY": "anchored"})
    live = make(settings, tmp_path / "s").build()
    from mactf.exceptions import ComponentLocked

    from mactf.components.kinds import Boundary

    with pytest.raises(ComponentLocked, match="E003"):
        live.slots.swap("TRAINER.POLICY", {"type": "anchored", "params": {"anchor_coefficient": 0.2}},
                        boundary=Boundary.EPISODE)


# --------------------------------------------------------------------- 18.8 retire, revive, purge
def _plan(name):
    return yaml.safe_load(plan_path(name).read_text(encoding="utf-8"))


def test_18_8_retire_revive_purge_through_the_engine(scope, settings, tmp_path):
    trainer = trainer_for("18.8", settings, tmp_path, peers=4)
    engine = TrainerEngine(settings, trainer=trainer)
    trainer.build()
    trainer.request_mutation(_plan("plan-retire"), source="cli")
    summary = engine.run()
    assert summary["status"] == "completed"
    assert "peer-3" in trainer.freeze.retired()
    assert "peer-3" not in trainer.episodes[-1].participants
    assert trainer.mutation_history()[0]["plan"]["label"] == "prune-peer-3"
    # revive it
    revived = trainer.mutate(_plan("plan-revive"))
    assert revived.revived == ("peer-3",)
    assert "peer-3" not in trainer.freeze.retired()
    # purge: refused while active, allowed once retired
    with pytest.raises(MutationError, match="E082"):
        trainer.preview_mutation(_plan("plan-purge"))
    trainer.retire_peer("peer-3")
    purged = trainer.mutate(_plan("plan-purge"))
    assert purged.purged == ("peer-3",)
    with pytest.raises(MutationError, match="E081.*peer-3|tombstone"):
        trainer.preview_mutation(_plan("plan-revive"))


def test_18_8_misbehaviours(scope, settings, tmp_path):
    trainer = trainer_for("18.8", settings, tmp_path, peers=4)
    trainer.build()
    with pytest.raises(MutationError, match="E082"):                      # PROTECTED
        trainer.preview_mutation([{"op": "retire", "path": "peer-1"}])
    retire = _plan("plan-retire")
    trainer.mutate(retire)
    again = trainer.mutate(retire)
    assert any("W040" in warning for warning in again.warnings)          # already retired: a no-op
    never = trainer.preview_mutation([{"op": "revive", "path": "peer-2"}])
    assert any("W041" in warning for warning in never.warnings)
    with pytest.raises(MutationError, match="E083"):
        trainer.preview_mutation([{"op": "add", "path": "peer-9", "like": "peer-ghost"}])
    with pytest.raises(MutationError, match="E081"):
        trainer.preview_mutation([{"op": "retire", "path": "nope"}])


@pytest.mark.parametrize("name", ["plan-retire", "plan-revive", "plan-purge"])
def test_the_plan_files_are_valid_plans(name):
    from mactf.training.mutation import MutationPlan

    plan = MutationPlan.parse(_plan(name))
    assert plan.label and plan.ops


# ----------------------------------------------------------------------------- the one named test
@pytest.mark.parametrize("number", NUMBERS)
def test_every_worked_example_runs(number, scope, settings, tmp_path):
    """Each example, through its shipped module and fragment, runs two epochs of a real engine."""
    entry = EXAMPLES[number]
    if entry["module"]:
        load_example(number)
    hierarchical = number in ("18.3", "18.4")
    peers = 4 if number == "18.8" else 3
    trainer = trainer_for(number, settings, tmp_path, hierarchy=hierarchical, peers=peers)
    summary = TrainerEngine(settings, trainer=trainer).run()
    assert summary["status"] == "completed" and summary["epochs_completed"] == 2
    assert all(torch.isfinite(peer.export_parameters()).all() for peer in trainer.peers.values())
    # the fragment's component is what ran
    manifest = trainer._component_manifest()
    if entry["module"]:
        recorded = {e["name"] for kind in manifest.values() for e in kind.values()}
        assert recorded & set(entry["names"]), (number, manifest)
    assert fragment(number) is not None and shipped.conf_path(number).exists()
