"""The function registry over the component model (8.2), the composed loss (7.8) and the new conf blocks."""

from __future__ import annotations

import copy

import pytest
import torch

from mactf import conf
from mactf.components import HUB, register_function
from mactf.components.loss_compliance import (
    CARRIES_COMPLIANCE,
    carries_compliance,
    ensure_compliance,
    iter_terms,
)
from mactf.exceptions import ComponentLocked, ContractError, ImproperlyConfigured, ValidationError
from mactf.rl.functions.loss import (
    CallableLoss,
    ChainedLoss,
    ComplianceFloorLoss,
    CompositeLoss,
    EntropyBonus,
    KLLoss,
    PolicyGradientLoss,
    ValueRegressionLoss,
    WeightedSumLoss,
)
from mactf.rl.functions.registry import (
    ALIASES,
    CAPABILITY_ORDER,
    FUNCTION_TYPES,
    build_function,
    function_capabilities,
    resolve_family,
    unregister_function,
)
from mactf.rl.policy import make_policy

from p2_support import ClippedAdvantage, HuberValueLoss


def _loss_batch(floor: float = -0.5):
    generator = torch.Generator().manual_seed(9)
    logits = torch.randn(4, 3, generator=generator, requires_grad=True)
    return {
        "policy": make_policy(logits), "reference": make_policy(torch.randn(4, 3, generator=generator)),
        "actions": torch.randint(0, 3, (4,), generator=generator), "advantages": torch.randn(4, generator=generator),
        "values": torch.randn(4, 1, generator=generator, requires_grad=True),
        "value_targets": torch.randn(4, 1, generator=generator),
        "compliance_floor": torch.tensor(floor), "void_mask": torch.zeros(4, dtype=torch.bool),
    }


# -- WeightedSumLoss -------------------------------------------------------------------------------------------
def test_weighted_sum_is_a_weighted_sum_of_registered_terms():
    batch = _loss_batch()
    terms = [PolicyGradientLoss(), ValueRegressionLoss(), KLLoss(), EntropyBonus()]
    loss = WeightedSumLoss(terms, weights=[1.0, 0.5, 0.02, 0.01])
    total, parts = loss.compute(batch)
    expected = sum(w * float(t.compute(batch)[0]) for w, t in zip([1.0, 0.5, 0.02, 0.01], terms))
    assert float(total) == pytest.approx(expected, rel=1e-6)
    assert parts.total == pytest.approx(expected, rel=1e-6)
    assert parts.extras["term:value_regression"] == pytest.approx(0.5 * float(ValueRegressionLoss().compute(batch)[0]))
    total.backward()                                        # differentiable end to end


def test_weighted_sum_builds_from_a_nested_spec_and_validates_every_term():
    loss = build_function({"type": "weighted_sum", "terms": [{"type": "policy_gradient", "clip_epsilon": 0.1},
                                                              {"type": "kl"}], "weights": [1.0, 0.1]},
                          family="loss")
    assert isinstance(loss, WeightedSumLoss) and loss.weights == [1.0, 0.1]
    assert loss.terms[0].clip_epsilon == 0.1
    assert loss.describe()["terms"][1]["type"] == "kl" and loss.describe()["weights"] == [1.0, 0.1]
    assert loss.required_fields == PolicyGradientLoss.required_fields | KLLoss.required_fields
    with pytest.raises(ValidationError, match="does not accept"):
        build_function({"type": "weighted_sum", "terms": [{"type": "kl", "nope": 1}]}, family="loss")
    assert WeightedSumLoss.from_spec({"type": "weighted_sum", "terms": [KLLoss()]}).weights == [1.0]


@pytest.mark.parametrize("terms,weights,message", [
    ([], None, "at least one term"),
    ([KLLoss()], [1.0, 2.0], "1 terms but 2 weights"),
    ([KLLoss()], ["x"], "not a number"),
    ([KLLoss()], [float("nan")], "not finite"),
    ([KLLoss()], [True], "not a number"),
    ([object()], None, "has no compute"),
])
def test_weighted_sum_refuses_nonsense(terms, weights, message):
    with pytest.raises(ValidationError, match=message):
        WeightedSumLoss(terms, weights=weights)


def test_weighted_sum_rejects_another_types_spec():
    with pytest.raises(ValidationError, match="cannot build type 'kl'"):
        WeightedSumLoss.from_spec({"type": "kl", "terms": [KLLoss()]})


def test_weighted_sum_reports_the_void_evidence_the_step_guard_reads():
    batch = _loss_batch()
    total, parts = WeightedSumLoss([CompositeLoss()]).compute(batch)
    assert parts.extras["advantage_mean"] == pytest.approx(float(batch["advantages"].mean()), rel=1e-6)


# -- ensure_compliance -------------------------------------------------------------------------------------------
def test_carries_compliance_walks_composed_losses():
    assert CARRIES_COMPLIANCE == {"compliance_floor", "composite"}
    assert carries_compliance(CompositeLoss())
    assert carries_compliance(ChainedLoss([KLLoss(), WeightedSumLoss([ComplianceFloorLoss()])]))
    assert not carries_compliance(WeightedSumLoss([KLLoss(), EntropyBonus()]))
    assert not carries_compliance(CallableLoss(lambda batch: batch["values"].mean()))
    assert [type(t).__name__ for t in iter_terms(ChainedLoss([KLLoss()]))] == ["ChainedLoss", "KLLoss"]


def test_ensure_compliance_require_extends_enforce_refuses_off_leaves():
    plain = CallableLoss(lambda batch: batch["values"].mean(), batch_keys=["values"])
    extended = ensure_compliance(plain, mode="require", coefficient=2.0)
    assert isinstance(extended, WeightedSumLoss) and extended.terms[0] is plain
    assert extended.terms[1].coefficient == 2.0
    again = ensure_compliance(extended, mode="require", coefficient=2.0)
    assert again is extended                                                # idempotent
    assert ensure_compliance(plain, mode="off", coefficient=1.0) is plain
    with pytest.raises(ValidationError, match="compliance_floor"):
        ensure_compliance(plain, mode="enforce", coefficient=1.0)
    assert ensure_compliance(CompositeLoss(), mode="enforce", coefficient=1.0) is not None
    with pytest.raises(ValidationError, match="off, require or enforce"):
        ensure_compliance(plain, mode="sometimes", coefficient=1.0)


def test_the_compliance_charge_is_zero_unless_the_floor_is_breached():
    plain = CallableLoss(lambda batch: batch["values"].mean(), batch_keys=["values"])
    extended = ensure_compliance(plain, mode="require", coefficient=1.0)
    batch = _loss_batch()
    clean = extended.compute({**batch, "compliance_floor": torch.tensor(0.5)})
    breached = extended.compute({**batch, "compliance_floor": torch.tensor(-2.0)})
    assert clean[1].compliance == 0.0 and breached[1].compliance == pytest.approx(2.0)
    assert float(breached[0]) == pytest.approx(float(clean[0]) + 2.0, rel=1e-6)


# -- the registry (8.2) ----------------------------------------------------------------------------------------------
def test_capabilities_keep_their_old_keys_and_gain_the_new_ones():
    catalog = function_capabilities()
    assert tuple(catalog) == CAPABILITY_ORDER
    entry = next(item for item in catalog["loss"] if item["type"] == "weighted_sum")
    assert {"type", "class", "kwargs", "origin", "version", "mutability"} <= set(entry)
    assert entry["origin"] == "builtin" and entry["mutability"] == "step" and entry["kwargs"] == ["terms", "weights"]
    rollups = {item["type"]: item for item in catalog["rollup"]}
    assert len(rollups) == 7 and rollups["kappa_weighted"]["alias_of"] == "kappa"
    assert "alias_of" not in rollups["kappa"] and ALIASES == {("rollup", "kappa_weighted"): "kappa"}


def test_an_overlay_registration_is_visible_to_the_catalog_and_resolution_then_gone(component_overlay):
    register_function("advantage", "clipped", ClippedAdvantage, scope="run", version="2")
    entry = next(item for item in function_capabilities()["advantage"] if item["type"] == "clipped")
    assert entry["origin"] == "project" and entry["version"] == "2" and entry["kwargs"] == ["clip"]
    assert resolve_family("clipped") == "advantage"
    assert build_function({"type": "clipped", "clip": 2.0}).clip == 2.0
    assert build_function({"type": "clipped", "params": {"clip": 3.0}}, family="advantage").clip == 3.0
    assert build_function("clipped" and {"type": "clipped"}, family="advantage").clip == 5.0


def test_after_the_overlay_the_registration_is_gone():
    with pytest.raises(ValidationError, match="unknown function type 'clipped'"):
        resolve_family("clipped")


def test_the_historical_error_texts_survive(component_overlay):
    with pytest.raises(ValidationError, match="requires a 'type' key"):
        build_function({})
    with pytest.raises(ValidationError, match="unknown function family"):
        build_function({"type": "x"}, family="nope")
    with pytest.raises(ValidationError, match="unknown loss type"):
        build_function({"type": "zzz"}, family="loss")
    with pytest.raises(ValidationError, match="does not accept"):
        build_function({"type": "kl", "nope": 1}, family="loss")
    with pytest.raises(ValidationError, match="ambiguous between families"):
        build_function({"type": "callable"})
    with pytest.raises(ValidationError, match="'import' spec cannot name its own family"):
        build_function({"import": "p2_support:ClippedAdvantage"})


def test_an_import_spec_registers_into_the_run_overlay(component_overlay):
    built = build_function({"import": "p2_support:ClippedAdvantage", "clip": 1.5}, family="advantage")
    assert built.clip == 1.5
    assert HUB.registry("rl_function.advantage").get("clipped_advantage").origin.value == "runtime"


def test_unregister_function_removes_a_custom_one_and_refuses_a_builtin(component_overlay):
    register_function("advantage", "clipped", ClippedAdvantage, scope="run")
    unregister_function("advantage", "clipped", scope="run")
    assert not HUB.registry("rl_function.advantage").has("clipped")
    with pytest.raises(ComponentLocked):
        unregister_function("advantage", "gae")


def test_inject_reaches_a_function_that_declares_it(component_overlay):
    class NeedsHost:
        def __init__(self, *, host=None, scale=1.0):
            self.host, self.scale = host, scale

        def __call__(self, module):
            return torch.zeros(())

    register_function("regularization", "needs_host", NeedsHost, scope="run")
    built = build_function({"type": "needs_host", "scale": 2.0}, family="regularization", inject={"host": "H"})
    assert built.host == "H" and built.scale == 2.0
    with pytest.raises(Exception, match="E011|supplied by the framework"):
        build_function({"type": "needs_host", "host": "mine"}, family="regularization", inject={"host": "H"})


def test_the_function_tables_are_still_the_live_views():
    assert FUNCTION_TYPES["loss"]["weighted_sum"] is WeightedSumLoss
    assert "weighted_sum" in HUB.registry("rl_function.loss").names()


# -- conf: the RL block ------------------------------------------------------------------------------------------------
def _validated(**blocks):
    values = copy.deepcopy(conf.DEFAULTS)
    for name, value in blocks.items():
        values[name] = value
    conf.validate_settings(conf.Settings(values))


def test_the_rl_default_block_is_inert():
    assert conf.DEFAULTS["RL"] == {"UNCONSUMED": "error", "STRICT_GLOBS": True, "VERIFY": "custom"}
    _validated()                                                      # and valid


@pytest.mark.parametrize("block,message", [
    ({"BOGUS": 1}, "RL has unknown key"),
    ({"UNCONSUMED": "ignore"}, "RL.UNCONSUMED"),
    ({"VERIFY": "maybe"}, "RL.VERIFY"),
    ({"STRICT_GLOBS": "yes"}, "RL.STRICT_GLOBS must be a boolean"),
    ({"MIGRATE": [1]}, "RL.MIGRATE must map"),
    ({"MIGRATE": {"NOPE": "reset"}}, "RL.MIGRATE.NOPE: unknown family"),
    ({"MIGRATE": {"LOSS": "sideways"}}, "RL.MIGRATE.LOSS must be"),
    ({"LOSS": 3}, "RL.LOSS: a spec is a string or a mapping"),
    ({"LOSS": {"params": {}}}, "exactly one of 'type' and 'import'"),
    ({"LOSS": {"type": "x", "import": "a:B"}}, "exactly one of 'type' and 'import'"),
    ({"LOSS": {"type": "x", "params": [1]}}, "'params' must be a mapping"),
    ({"LOSS": ""}, "an empty string"),
    ({"OVERRIDES": [1]}, "RL.OVERRIDES must map FAMILY"),
    ({"OVERRIDES": {"NOPE": {}}}, "RL.OVERRIDES.NOPE: unknown family"),
    ({"OVERRIDES": {"ROLLUP": {"a": "mean"}}}, "E042"),
    ({"OVERRIDES": {"LOSS": [1]}}, "RL.OVERRIDES.LOSS must map glob"),
    ({"OVERRIDES": {"LOSS": {" ": "kl"}}}, "a glob cannot be empty"),
    ({"OVERRIDES": {"LOSS": {"a": {"nope": 1}}}}, "exactly one of 'type' and 'import'"),
])
def test_the_rl_block_shape_is_validated_without_importing_torch(block, message):
    with pytest.raises(ImproperlyConfigured, match=message):
        _validated(RL=block)


def test_a_valid_rl_block_loads():
    _validated(RL={"ADVANTAGE": {"type": "clipped", "params": {"clip": 3.0}}, "LOSS": "composite",
                   "MIGRATE": {"loss": "carry"}, "OVERRIDES": {"ADVANTAGE": {"peer-*": {"type": "gae"}}},
                   "UNCONSUMED": "warn", "VERIFY": "all", "STRICT_GLOBS": False})


def test_rl_must_be_a_mapping():
    with pytest.raises(ImproperlyConfigured, match="RL must be a mapping"):
        _validated(RL=3)


# -- conf: OPTIMIZER and the TRAINER keys -----------------------------------------------------------------------------------
def test_optimizer_default_block_has_the_plan_keys():
    block = conf.DEFAULTS["OPTIMIZER"]
    assert block["OVERRIDES"] == {} and block["MIGRATE"] == "reset" and block["VERIFY"] == "custom"
    assert block["STRICT_GLOBS"] is True and block["TYPE"] == "torch"        # the legacy keys unchanged


@pytest.mark.parametrize("block,message", [
    ({"TYPE": ""}, "TYPE must be a non-empty string"),
    ({"MIGRATE": "sideways"}, "OPTIMIZER.MIGRATE"),
    ({"VERIFY": "maybe"}, "OPTIMIZER.VERIFY"),
    ({"STRICT_GLOBS": 1}, "STRICT_GLOBS must be a boolean"),
    ({"OVERRIDES": [1]}, "OPTIMIZER.OVERRIDES must map"),
    ({"OVERRIDES": {"": {}}}, "keys must be non-empty strings"),
    ({"OVERRIDES": {"a": 3}}, "must be a mapping"),
    ({"OVERRIDES": {"a": {"OVERRIDES": {}}}}, "cannot contain OVERRIDES"),
    ({"OVERRIDES": {"a": {"LR": -1}}}, r"OVERRIDES\['a'\].LR must be >= 0"),
    ({"OVERRIDES": {"a": {"GRAD_ACCUMULATION": 0}}}, "GRAD_ACCUMULATION must be >= 1"),
    ({"OVERRIDES": {"a": {"PARAMS": [1]}}}, "PARAMS must be a mapping"),
    ({"OVERRIDES": {"a": {"SCHEDULER_KWARGS": [1]}}}, "SCHEDULER_KWARGS must be a mapping"),
    ({"GRAD_ACCUMULATION": 0}, "OPTIMIZER.GRAD_ACCUMULATION must be >= 1"),
    ({"MAX_GRAD_NORM": -1}, "MAX_GRAD_NORM must be >= 0"),
])
def test_optimizer_block_shape_is_validated(block, message):
    merged = {**copy.deepcopy(conf.DEFAULTS["OPTIMIZER"]), **block}
    with pytest.raises(ImproperlyConfigured, match=message):
        _validated(OPTIMIZER=merged)


def test_any_registered_name_is_a_valid_type_at_load():
    merged = {**copy.deepcopy(conf.DEFAULTS["OPTIMIZER"]), "TYPE": "Lion", "PARAMS": {"betas": [0.9, 0.99]},
              "OVERRIDES": {"peer-*": {"TYPE": "null"}}}
    _validated(OPTIMIZER=merged)


@pytest.mark.parametrize("trainer,message", [
    ({"POLICY": 3}, "TRAINER.POLICY"),
    ({"POLICY": {"params": {}}}, "needs 'type'"),
    ({"POLICY": "nonsense"}, "TRAINER.POLICY must be one of"),
    ({"ALLOW_UNSAFE_POLICY": "yes"}, "must be a boolean"),
    ({"LOSS_COMPLIANCE": "never"}, "TRAINER.LOSS_COMPLIANCE"),
    ({"COMPLIANCE_COEFFICIENT": -1}, "COMPLIANCE_COEFFICIENT"),
    ({"STEP_GUARD": 3}, "STEP_GUARD must be a mapping"),
    ({"STEP_GUARD": {"NOPE": 1}}, "unknown TRAINER.STEP_GUARD key"),
    ({"STEP_GUARD": {"MODE": "loud"}}, "STEP_GUARD.MODE"),
    ({"STEP_GUARD": {"SNAPSHOT": "x"}}, "STEP_GUARD.SNAPSHOT"),
    ({"STEP_GUARD": {"MAX_CONSECUTIVE": 0}}, "MAX_CONSECUTIVE"),
    ({"STEP_GUARD": {"GRAD_NORM_CEILING": 0}}, "GRAD_NORM_CEILING"),
    ({"STEP_GUARD": {"MAX_GRAD_NORM": -2}}, "MAX_GRAD_NORM"),
])
def test_the_trainer_policy_keys_are_validated(trainer, message):
    merged = {**copy.deepcopy(conf.DEFAULTS["TRAINER"]), **trainer}
    with pytest.raises(ImproperlyConfigured, match=message):
        _validated(TRAINER=merged)


def test_valid_trainer_policy_keys_load():
    merged = {**copy.deepcopy(conf.DEFAULTS["TRAINER"]), "POLICY": {"type": "anchored", "params": {"a": 1}},
              "ALLOW_UNSAFE_POLICY": False, "LOSS_COMPLIANCE": "enforce", "COMPLIANCE_COEFFICIENT": 2,
              "STEP_GUARD": {"mode": "warn", "max_consecutive": 5, "max_grad_norm": None}}
    _validated(TRAINER=merged, COMPONENTS={"MODULES": ["acme.policies"]})


def test_torch_is_not_imported_by_the_validators():
    import subprocess
    import sys

    code = (
        "import sys, copy\n"
        "from mactf import conf\n"
        "v = copy.deepcopy(conf.DEFAULTS)\n"
        "v['RL'] = {'ADVANTAGE': {'type': 'clipped'}, 'OVERRIDES': {'LOSS': {'a': 'kl'}}}\n"
        "v['OPTIMIZER']['OVERRIDES'] = {'a': {'TYPE': 'Lion'}}\n"
        "v['TRAINER']['STEP_GUARD'] = {'MODE': 'warn'}\n"
        "conf.validate_settings(conf.Settings(v))\n"
        "assert 'torch' not in sys.modules, 'validation imported torch'\n"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
