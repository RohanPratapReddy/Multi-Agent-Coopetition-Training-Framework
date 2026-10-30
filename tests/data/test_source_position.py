from __future__ import annotations

import json

import torch

from mactf.data.sources import JsonlTaskSource, MixedTaskSource, SyntheticTaskSource


def _synthetic():
    return SyntheticTaskSource(state_dim=4, action_dim=3, batch_size=5, seed=7)


def test_synthetic_position_roundtrip():
    a = _synthetic()
    for epoch in range(3):
        a.next(epoch)
    state = a.state_dict()
    b = _synthetic()
    b.load_state_dict(state)
    assert torch.equal(a.next(3).states, b.next(3).states)


def test_synthetic_rejects_other_seed():
    other = SyntheticTaskSource(state_dim=4, action_dim=3, batch_size=5, seed=8)
    try:
        other.load_state_dict(_synthetic().state_dict())
    except ValueError as error:
        assert "seed" in str(error)
    else:  # pragma: no cover
        raise AssertionError("seed mismatch must be rejected")


def test_jsonl_and_mixed_position_roundtrip(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text(
        "\n".join(json.dumps({"id": f"r{i}", "prompt": f"p{i}"}) for i in range(10)),
        encoding="utf-8",
    )

    def build():
        jsonl = JsonlTaskSource([path], state_dim=4, action_dim=3, batch_size=3, seed=1)
        return MixedTaskSource([(jsonl, 1.0), (_synthetic(), 1.0)])

    a = build()
    for _ in range(5):
        a.next(0)
    b = build()
    b.load_state_dict(a.state_dict())
    for _ in range(4):
        left, right = a.next(0), b.next(0)
        assert torch.equal(left.states, right.states)
        assert left.id == right.id
