"""``specs.trainer(distillation=...)`` shape and server parity (doc 14 section 19.3)."""

from __future__ import annotations

import pytest

from mactf.client import specs
from mactf.contrib.api.serializers import TrainerSpecSerializer


def validated(payload: dict) -> dict:
    serializer = TrainerSpecSerializer(data=payload)
    assert serializer.is_valid(), serializer.errors
    return serializer.validated_data


def test_trainer_spec_without_distillation_has_no_such_key():
    assert "distillation" not in specs.trainer(num_peers=2)


def test_true_and_false_become_an_enabled_mapping():
    assert specs.trainer(distillation=True)["distillation"] == {"ENABLED": True}
    assert specs.trainer(distillation=False)["distillation"] == {"ENABLED": False}


def test_a_mapping_is_copied_not_aliased():
    block = {"TEACHERS": [{"name": "t1"}], "TAU": 0.1}
    produced = specs.trainer(distillation=block)["distillation"]
    assert produced == block
    assert produced is not block
    block["TAU"] = 0.9
    assert produced["TAU"] == 0.1


def test_none_is_dropped_so_the_server_defers_to_settings():
    assert "distillation" not in specs.trainer(distillation=None)


def test_hierarchy_and_num_peers_stay_exclusive_with_distillation_present():
    with pytest.raises(ValueError, match="mutually exclusive"):
        specs.trainer(hierarchy={"levels": []}, num_peers=2, distillation=True)


@pytest.mark.parametrize("value", [True, False])
def test_the_normalised_boolean_is_accepted_by_the_server_serializer(value):
    data = validated(specs.trainer(num_peers=2, distillation=value))
    assert data["distillation"]["ENABLED"] is value


def test_a_full_block_round_trips_through_the_server_serializer():
    block = {
        "ENABLED": True,
        "TEACHERS": [{"name": "t1", "kind": "module", "factory": "torch.nn.Linear"}],
        "TAU": 0.2,
        "SIGMA_MODE": "uniform",
        "CONFORMANCE": {"MODE": "declared", "TRUST_DECAY": 0.4},
    }
    data = validated(specs.trainer(num_peers=2, distillation=block))
    assert data["distillation"]["TAU"] == pytest.approx(0.2)
    assert data["distillation"]["CONFORMANCE"]["MODE"] == "declared"
