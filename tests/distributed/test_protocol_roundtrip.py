"""Every protocol dataclass survives ``to_dict``/``from_dict`` unchanged."""

from __future__ import annotations

import json

import pytest

from mactf.distributed.protocol import (
    ParameterExchange,
    TaskAssignment,
    TaskResult,
    WorkerHeartbeat,
    WorkerRegistration,
)

SAMPLES = [
    WorkerRegistration("w1", "host-a", {"device": "cpu", "dtype": "float32", "torch": "2.x"}, max_peers=3),
    WorkerHeartbeat("w1", 4, ("p0", "p1"), 0.5, 123456789),
    TaskAssignment(
        "r1", 2, "t1", (8, 4), (8,), ("p0", "p1"), "QUJD", 600.0, checksum="ab12",
        config={"signals": {"p0": 0.25}, "void": {"p0": False}, "loss": {"kl_coef": 0.01}},
    ),
    TaskResult("t1", "w1", ("p0",), "QUJD", "ab12", {"p0.total": 1.5, "duration": 0.2}, True, None, epoch=2, run_id="r1"),
    TaskResult("t2", "w1", ("p0",), "", "", {}, False, "boom"),
    ParameterExchange("r1", 2, "QUJD", "ab12", "REVG", 1.25, reference_checksum="cd34", reset_optimizers=False),
    ParameterExchange("r1", 3, "QUJD", "ab12", None, 0.5),
]


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: type(s).__name__)
def test_roundtrip_through_json(sample):
    wire = json.loads(json.dumps(sample.to_dict()))
    assert type(sample).from_dict(wire) == sample


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: type(s).__name__)
def test_unknown_keys_are_ignored_and_required_ones_are_named(sample):
    wire = sample.to_dict()
    assert type(sample).from_dict({**wire, "future": 1}) == sample
    required = next(iter(wire))
    broken = {k: v for k, v in wire.items() if k != required}
    with pytest.raises(KeyError, match=type(sample).__name__):
        type(sample).from_dict(broken)


def test_dataclasses_are_frozen():
    with pytest.raises(Exception):
        SAMPLES[0].worker_id = "other"  # type: ignore[misc]
