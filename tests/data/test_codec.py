"""Codec validation: a dataset typo must fail the run, not change it."""

from __future__ import annotations

import pytest

from mactf.data.codec import TaskCodec, TaskRecord
from mactf.exceptions import DataError


def test_unknown_key_raises():
    with pytest.raises(DataError) as captured:
        TaskCodec.decode({"id": "r1", "promt": "typo"}, line=3, path="dataset.jsonl")
    message = str(captured.value)
    assert "unknown key" in message
    assert "promt" in message
    assert "dataset.jsonl:3" in message


def test_missing_id_raises():
    with pytest.raises(DataError) as captured:
        TaskCodec.decode({"prompt": "no id"}, line=1, path="d.jsonl")
    assert "non-empty string 'id'" in str(captured.value)


def test_empty_id_raises():
    with pytest.raises(DataError):
        TaskCodec.decode({"id": "  "}, line=1, path="d.jsonl")


def test_states_without_targets_raises():
    with pytest.raises(DataError) as captured:
        TaskCodec.decode({"id": "r", "states": [[1.0, 2.0]]}, line=7, path="d.jsonl")
    assert "both be present or both absent" in str(captured.value)


def test_length_mismatch_raises():
    payload = {"id": "r", "states": [[1.0], [2.0]], "targets": [0]}
    with pytest.raises(DataError) as captured:
        TaskCodec.decode(payload, line=9, path="d.jsonl")
    assert "must match" in str(captured.value)


def test_ragged_rows_raise():
    payload = {"id": "r", "states": [[1.0, 2.0], [3.0]], "targets": [0, 1]}
    with pytest.raises(DataError) as captured:
        TaskCodec.decode(payload, line=2, path="d.jsonl")
    assert "same length" in str(captured.value)


def test_target_out_of_range_raises():
    payload = {"id": "r", "states": [[1.0]], "targets": [5]}
    with pytest.raises(DataError) as captured:
        TaskCodec.decode(payload, line=2, path="d.jsonl", action_dim=4)
    assert "action_dim" in str(captured.value)


def test_non_finite_state_raises():
    payload = {"id": "r", "states": [[float("nan")]], "targets": [0]}
    with pytest.raises(DataError):
        TaskCodec.decode(payload, line=2, path="d.jsonl")


def test_error_carries_location_and_id():
    with pytest.raises(DataError) as captured:
        TaskCodec.decode({"id": "rec-9", "targets": "no"}, line=42, path="d.jsonl")
    error = captured.value
    assert error.path == "d.jsonl"
    assert error.line == 42
    assert error.record_id == "rec-9"
    assert error.key == "targets"


def test_claim_record_folds_into_prompt():
    record = TaskCodec.decode(
        {"id": "t1", "claim": "2 + 2 = 4", "evidence": "arithmetic", "verified": True},
        line=1,
        path="truth.jsonl",
    )
    assert record.prompt == "2 + 2 = 4"
    assert record.evidence == "arithmetic"
    assert record.verified is True
    assert record.metadata["claim"] == "2 + 2 = 4"


def test_interaction_record_keeps_expects():
    record = TaskCodec.decode(
        {"id": "i1", "stage": "critique", "peers": 3, "expects": ["R11", "R4"]},
        line=1,
        path="interaction.jsonl",
    )
    assert record.metadata["stage"] == "critique"
    assert record.metadata["peers"] == 3
    assert record.metadata["expects"] == ["R11", "R4"]


def test_round_trip():
    record = TaskRecord(
        id="r",
        prompt="p",
        states=((1.0, 2.0), (3.0, 4.0)),
        targets=(0, 1),
        expected="admitting_ignorance",
        metadata={"k": "v"},
    )
    encoded = TaskCodec.encode(record)
    assert TaskCodec.decode(encoded, line=1, path="d.jsonl") == record
