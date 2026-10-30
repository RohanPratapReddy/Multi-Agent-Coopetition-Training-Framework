"""``TrainerSpecSerializer`` and ``trainer_kwargs`` for the distillation block (doc 14 section 19.3)."""

from __future__ import annotations

import pytest

from mactf.contrib.api.serializers import (
    DistillationConformanceSerializer,
    DistillationSpecSerializer,
    TrainerSpecSerializer,
)
from mactf.contrib.api.views.scorers import trainer_kwargs
from mactf.distill import DistillationConfig

TEACHER = {
    "name": "t1",
    "kind": "module",
    "factory": "torch.nn.Linear",
    "kwargs": {"in_features": 4, "out_features": 3},
}


def run(payload: dict):
    serializer = TrainerSpecSerializer(data=payload)
    return serializer.is_valid(), serializer


def test_absent_block_is_none():
    ok, serializer = run({"num_peers": 2})
    assert ok
    assert serializer.validated_data["distillation"] is None


def test_a_valid_block_passes():
    ok, serializer = run(
        {"num_peers": 2, "distillation": {"ENABLED": True, "TEACHERS": [TEACHER], "TAU": 0.2}}
    )
    assert ok, serializer.errors
    assert serializer.validated_data["distillation"]["TAU"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("block", "field"),
    [
        ({"TAUU": 0.1}, "TAUU"),
        ({"SIGMA_MODE": "bogus"}, "SIGMA_MODE"),
        ({"OUT_DIM": 0}, "OUT_DIM"),
        ({"PEER_MARGIN": 2.0}, "PEER_MARGIN"),
        ({"SIGMA_CAP": 1.5}, "SIGMA_CAP"),
        ({"WARMUP_STEPS": -1}, "WARMUP_STEPS"),
        ({"BETA": -0.5}, "BETA"),
    ],
)
def test_bad_fields_are_rejected_at_the_edge(block, field):
    ok, serializer = run({"num_peers": 2, "distillation": block})
    assert not ok
    assert field in str(serializer.errors)


def test_unknown_conformance_key_is_rejected_at_the_edge():
    ok, serializer = run(
        {"num_peers": 2, "distillation": {"TEACHERS": [TEACHER], "CONFORMANCE": {"TRUST_DECAYY": 1}}}
    )
    assert not ok
    assert "TRUST_DECAYY" in str(serializer.errors)


def test_conformance_mode_and_bounds_are_checked():
    nested = DistillationConformanceSerializer(data={"MODE": "bogus"})
    assert not nested.is_valid()
    nested = DistillationConformanceSerializer(data={"TRUST_MIN": 2.0})
    assert not nested.is_valid()


def test_cross_field_rules_run_at_the_edge():
    ok, serializer = run(
        {
            "num_peers": 2,
            "distillation": {
                "TEACHERS": [TEACHER, {**TEACHER, "name": "t2"}],
                "SIGMA_FLOOR": 0.8,
            },
        }
    )
    assert not ok
    assert "exceeds 1.0" in str(serializer.errors)


def test_teacher_presence_is_left_to_the_settings_block():
    """An enabled block with no teachers may be completed by the project settings."""
    assert DistillationSpecSerializer(data={"ENABLED": True}).is_valid()


def test_trainer_kwargs_forwards_the_block_without_none_entries(settings):
    ok, serializer = run(
        {
            "num_peers": 2,
            "distillation": {"ENABLED": True, "TEACHERS": [TEACHER], "TAU": 0.2},
        }
    )
    assert ok
    kwargs = trainer_kwargs(serializer.validated_data, settings)
    block = kwargs["distillation"]
    assert block["TAU"] == pytest.approx(0.2)
    assert block["ENABLED"] is True
    assert None not in block.values()
    assert "SIGMA_MODE" not in block  # unset means "defer", so it never appears


def test_trainer_kwargs_drops_none_inside_conformance(settings):
    ok, serializer = run(
        {
            "num_peers": 2,
            "distillation": {"TEACHERS": [TEACHER], "CONFORMANCE": {"MODE": "declared"}},
        }
    )
    assert ok
    block = trainer_kwargs(serializer.validated_data, settings)["distillation"]
    assert block["CONFORMANCE"] == {"MODE": "declared"}


def test_trainer_kwargs_omits_distillation_when_absent(settings):
    ok, serializer = run({"num_peers": 2})
    assert ok
    assert "distillation" not in trainer_kwargs(serializer.validated_data, settings)


def test_a_forwarded_block_resolves_to_the_same_config_as_the_sdk_mapping(settings):
    block = {"TEACHERS": [TEACHER], "TAU": 0.2, "CONFORMANCE": {"TRUST_DECAY": 0.4}}
    ok, serializer = run({"num_peers": 2, "distillation": block})
    assert ok
    forwarded = trainer_kwargs(serializer.validated_data, settings)["distillation"]
    from_wire = DistillationConfig.from_spec(forwarded, settings=settings)
    direct = DistillationConfig.from_spec(block, settings=settings)
    assert from_wire == direct
    assert from_wire.enabled is True


def test_an_enabled_false_block_forwards_as_off(settings):
    ok, serializer = run({"num_peers": 2, "distillation": {"ENABLED": False}})
    assert ok
    forwarded = trainer_kwargs(serializer.validated_data, settings)["distillation"]
    assert DistillationConfig.from_spec(forwarded, settings=settings).enabled is False
