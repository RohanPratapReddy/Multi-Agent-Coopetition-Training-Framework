"""The DISTILLATION settings block and the ``distillation=`` resolution table (doc 14 section 10)."""

from __future__ import annotations

import copy
import json

import pytest

from distill_support import distillation_block, teacher_spec
from mactf import conf
from mactf.distill.config import (
    CONFORMANCE_KEYS,
    DISTILLATION_DEFAULTS,
    DISTILLATION_KEYS,
    MAX_TEACHERS,
    DistillationConfig,
    validate_block,
)
from mactf.exceptions import ImproperlyConfigured, ValidationError


def with_block(settings, **block):
    merged = copy.deepcopy(DISTILLATION_DEFAULTS)
    merged.update(block)
    settings["DISTILLATION"] = merged
    return settings


# -- defaults -----------------------------------------------------------------------
def test_defaults_are_off_and_valid(settings):
    block = settings["DISTILLATION"]
    assert block["ENABLED"] is False
    assert block["TEACHERS"] == []
    assert block["CONFORMANCE"]["MODE"] == "hybrid"
    conf.validate_settings(settings)


def test_defaults_are_not_shared_between_settings_objects(settings):
    settings["DISTILLATION"]["TEACHERS"].append({"name": "leak"})
    assert DISTILLATION_DEFAULTS["TEACHERS"] == []


def test_keys_are_derived_from_the_defaults():
    assert DISTILLATION_KEYS == frozenset(DISTILLATION_DEFAULTS)
    assert CONFORMANCE_KEYS == frozenset(DISTILLATION_DEFAULTS["CONFORMANCE"])


# -- validate_settings --------------------------------------------------------------
@pytest.mark.parametrize(
    ("block", "message"),
    [
        ({"ENABLED": True}, "TEACHERS is empty"),
        ({"BOGUS": 1}, "does not accept 'BOGUS'"),
        ({"SIGMA_MODE": "bogus"}, "SIGMA_MODE"),
        ({"TAU": 0.0}, "TAU must be strictly positive"),
        ({"TAU_MIN": 0.0}, "TAU_MIN"),
        ({"TAU_MIN": 0.5, "TAU_MAX": 0.1}, "TAU_MAX"),
        ({"BETA": -1.0}, "BETA"),
        ({"OUT_DIM": 0}, "OUT_DIM"),
        ({"OUT_DIM": 2.5}, "OUT_DIM"),
        ({"SIGMA_DECAY": 1.0}, "SIGMA_DECAY"),
        ({"SIGMA_CAP": 0.0}, "SIGMA_CAP must be in"),
        ({"SIGMA_FLOOR": 0.5, "SIGMA_CAP": 0.4}, "SIGMA_FLOOR must be within"),
        ({"LAMBDA_CD": -0.1}, "LAMBDA_CD"),
        ({"PEER_MARGIN": 1.5}, "PEER_MARGIN"),
        ({"WARMUP_STEPS": -1}, "WARMUP_STEPS"),
        ({"RAMP_STEPS": 1.5}, "RAMP_STEPS"),
        ({"VIEWS": []}, "VIEWS must be a mapping"),
        ({"VIEWS": {"*": "identity"}}, "must be a mapping"),
        ({"VIEWS": {"*": {"kind": "bogus"}}}, "accepted: identity"),
        ({"REGULARIZERS": "kl"}, "REGULARIZERS"),
        ({"CONFORMANCE": []}, "CONFORMANCE must be a mapping"),
    ],
)
def test_validate_settings_rejects_bad_blocks(settings, block, message):
    with_block(settings, **block)
    with pytest.raises((ImproperlyConfigured, ValidationError), match=message):
        conf.validate_settings(settings)


def test_all_zero_loss_weights_are_refused(settings):
    with_block(settings, LAMBDA_IND=0.0, LAMBDA_CD=0.0, LAMBDA_COOP=0.0)
    with pytest.raises(ImproperlyConfigured, match="train nothing"):
        conf.validate_settings(settings)


def test_a_non_mapping_block_is_refused(settings):
    settings["DISTILLATION"] = "on"
    with pytest.raises(ImproperlyConfigured, match="must be a mapping"):
        conf.validate_settings(settings)


def test_too_many_teachers_are_refused():
    teachers = [teacher_spec(f"t{i}") for i in range(MAX_TEACHERS + 1)]
    with pytest.raises(ImproperlyConfigured, match="configuration mistake"):
        validate_block({**DISTILLATION_DEFAULTS, "TEACHERS": teachers})


def test_sigma_bounds_must_admit_a_normalized_vector():
    two = [teacher_spec("a"), teacher_spec("b")]
    with pytest.raises(ImproperlyConfigured, match="exceeds 1.0"):
        validate_block({**DISTILLATION_DEFAULTS, "TEACHERS": two, "SIGMA_FLOOR": 0.6})
    with pytest.raises(ImproperlyConfigured, match="below 1.0"):
        validate_block({**DISTILLATION_DEFAULTS, "TEACHERS": two, "SIGMA_CAP": 0.4})


@pytest.mark.parametrize(
    ("conformance", "message"),
    [
        ({"TRUST_DECAYY": 0.1}, "TRUST_DECAYY"),
        ({"MODE": "bogus"}, "MODE must be"),
        ({"TRUST_INIT": 0.5, "TRUST_FLOOR": 0.8}, "TRUST_FLOOR <= TRUST_INIT"),
        ({"TRUST_MIN": 0.0}, "TRUST_MIN must be in"),
        ({"TRUST_FLOOR": 0.3, "TRUST_MIN": 0.1}, "below TRUST_FLOOR"),
        ({"TRUST_DECAY": 0.0}, "TRUST_DECAY"),
        ({"TRUST_RESTORE": 1.5}, "TRUST_RESTORE"),
        ({"AUDIT_ARTIFACTS": ["x"]}, "must map teacher name"),
        ({"AUDIT_ARTIFACTS": {"ghost": "x.json"}}, "not in DISTILLATION.TEACHERS"),
    ],
)
def test_conformance_block_is_validated(conformance, message):
    block = {
        **copy.deepcopy(DISTILLATION_DEFAULTS),
        "TEACHERS": [teacher_spec()],
        "CONFORMANCE": {**DISTILLATION_DEFAULTS["CONFORMANCE"], **conformance},
    }
    with pytest.raises(ImproperlyConfigured, match=message):
        validate_block(block)


def test_a_missing_audit_artifact_is_not_an_empty_audit(tmp_path):
    block = {
        **copy.deepcopy(DISTILLATION_DEFAULTS),
        "TEACHERS": [teacher_spec()],
        "CONFORMANCE": {
            **DISTILLATION_DEFAULTS["CONFORMANCE"],
            "AUDIT_ARTIFACTS": {"t1": str(tmp_path / "missing.json")},
        },
    }
    with pytest.raises(ImproperlyConfigured, match="not a file"):
        validate_block(block)
    artifact = tmp_path / "audit.json"
    artifact.write_text("{}", encoding="utf-8")
    block["CONFORMANCE"]["AUDIT_ARTIFACTS"] = {"t1": str(artifact)}
    validate_block(block)


def test_audit_mode_requires_every_teacher_to_have_an_auditor():
    block = {
        **copy.deepcopy(DISTILLATION_DEFAULTS),
        "TEACHERS": [teacher_spec()],
        "CONFORMANCE": {**DISTILLATION_DEFAULTS["CONFORMANCE"], "MODE": "audit"},
    }
    with pytest.raises(ImproperlyConfigured, match="declare no auditor"):
        validate_block(block)
    block["TEACHERS"] = [teacher_spec(audit={"kind": "guard"})]
    validate_block(block)
    block["CONFORMANCE"]["ENABLED"] = False
    block["TEACHERS"] = [teacher_spec()]
    validate_block(block)


# -- the resolution table -----------------------------------------------------------
def test_none_defers_to_the_settings_block(settings):
    assert DistillationConfig.from_spec(None, settings=settings).enabled is False
    with_block(settings, ENABLED=True, TEACHERS=[teacher_spec()], TAU=0.3)
    config = DistillationConfig.from_spec(None, settings=settings)
    assert config.enabled is True
    assert config.tau == pytest.approx(0.3)


def test_false_is_unconditional_off(settings):
    with_block(settings, ENABLED=True, TEACHERS=[teacher_spec()])
    config = DistillationConfig.from_spec(False, settings=settings)
    assert config.enabled is False


def test_false_does_not_validate_teachers(settings):
    config = DistillationConfig.from_spec(False, settings=settings)
    assert config.enabled is False and config.teachers == ()


def test_true_is_unconditional_on_and_needs_teachers(settings):
    with pytest.raises(ImproperlyConfigured, match="TEACHERS is empty"):
        DistillationConfig.from_spec(True, settings=settings)
    with_block(settings, ENABLED=False, TEACHERS=[teacher_spec()])
    assert DistillationConfig.from_spec(True, settings=settings).enabled is True


def test_a_mapping_without_enabled_means_on(settings):
    config = DistillationConfig.from_spec({"TEACHERS": [teacher_spec()]}, settings=settings)
    assert config.enabled is True


def test_a_mapping_with_enabled_false_is_off_whatever_the_settings_say(settings):
    with_block(settings, ENABLED=True, TEACHERS=[teacher_spec()])
    config = DistillationConfig.from_spec({"ENABLED": False, "TAU": 0.2}, settings=settings)
    assert config.enabled is False
    assert config.tau == pytest.approx(0.2)


def test_a_settings_enabled_never_leaks_under_an_explicit_mapping(settings):
    with_block(settings, ENABLED=False, TEACHERS=[teacher_spec()])
    assert DistillationConfig.from_spec({"TAU": 0.2}, settings=settings).enabled is True


def test_explicit_mapping_overrides_settings_and_defaults(settings):
    with_block(settings, TEACHERS=[teacher_spec()], TAU=0.3, BETA=2.0)
    config = DistillationConfig.from_spec({"TAU": 0.2}, settings=settings)
    assert config.tau == pytest.approx(0.2)
    assert config.beta == pytest.approx(2.0)
    assert config.out_dim == DISTILLATION_DEFAULTS["OUT_DIM"]
    assert len(config.teachers) == 1


def test_conformance_is_deep_merged(settings):
    with_block(
        settings,
        TEACHERS=[teacher_spec()],
        CONFORMANCE={**DISTILLATION_DEFAULTS["CONFORMANCE"], "TRUST_DECAY": 0.4},
    )
    config = DistillationConfig.from_spec({"CONFORMANCE": {"MODE": "declared"}}, settings=settings)
    assert config.conformance["MODE"] == "declared"
    assert config.conformance["TRUST_DECAY"] == pytest.approx(0.4)
    assert config.conformance["TRUST_MIN"] == pytest.approx(0.05)


def test_a_config_instance_passes_through(settings):
    config = DistillationConfig.from_spec({"TEACHERS": [teacher_spec()]}, settings=settings)
    assert DistillationConfig.from_spec(config, settings=settings) is config


def test_unknown_key_and_unknown_type_are_refused(settings):
    with pytest.raises(ValidationError, match="does not accept 'TAUU'"):
        DistillationConfig.from_spec({"TAUU": 0.1, "TEACHERS": [teacher_spec()]}, settings=settings)
    with pytest.raises(ValidationError, match="bool, a mapping"):
        DistillationConfig.from_spec("on", settings=settings)


def test_from_spec_does_not_mutate_its_inputs(settings):
    explicit = distillation_block(CONFORMANCE={"MODE": "declared"})
    snapshot = copy.deepcopy(explicit)
    before = copy.deepcopy(settings["DISTILLATION"])
    DistillationConfig.from_spec(explicit, settings=settings)
    assert explicit == snapshot
    assert settings["DISTILLATION"] == before


def test_as_dict_round_trips_and_is_json_safe(settings):
    config = DistillationConfig.from_spec(
        distillation_block(SIGMA_MODE="uniform", CONFORMANCE={"TRUST_DECAY": 0.25}),
        settings=settings,
    )
    payload = config.as_dict()
    json.dumps(payload)
    assert payload["ENABLED"] is True
    assert set(payload) == DISTILLATION_KEYS
    assert DistillationConfig.from_spec(payload, settings=settings) == config


def test_objective_kwargs_follow_the_config(settings):
    config = DistillationConfig.from_spec(
        distillation_block(TAU=0.2, LAMBDA_COOP=0.5, WARMUP_STEPS=3), settings=settings
    )
    kwargs = config.objective_kwargs()
    assert kwargs["tau"] == pytest.approx(0.2)
    assert kwargs["lambda_coop"] == pytest.approx(0.5)
    assert kwargs["warmup_steps"] == 3
