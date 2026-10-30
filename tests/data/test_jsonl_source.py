"""JSONL task sources: loading, determinism across processes, traceability."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mactf.data.sources import JsonlTaskSource, MixedTaskSource, SyntheticTaskSource
from mactf.exceptions import DataError

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "training" / "datasets"


@pytest.fixture()
def datasets(tmp_path):
    """The shipped three datasets copied into a temp directory."""
    target = tmp_path / "datasets"
    target.mkdir()
    for name in ("truth.jsonl", "ambiguity.jsonl", "interaction.jsonl"):
        (target / name).write_text(
            (DATASETS / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    return target


def _source(paths, **kwargs):
    options = dict(state_dim=8, action_dim=4, batch_size=16, seed=7)
    options.update(kwargs)
    return JsonlTaskSource(paths, **options)


def test_shipped_files_load(datasets):
    for name, expected in (
        ("truth.jsonl", 3),
        ("ambiguity.jsonl", 3),
        ("interaction.jsonl", 4),
    ):
        source = _source([datasets / name])
        line_count = sum(
            1
            for line in (datasets / name).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        assert line_count == expected
        assert len(source.records()) == expected


def test_record_count_matches_line_count(datasets):
    payload = b'{"id": "a"}\n{"id": "b"}\n'
    path = datasets / "extra.jsonl"
    path.write_bytes(payload)
    source = _source([path])
    assert len(source.records()) == payload.count(b"\n")


def test_metadata_carries_traceability(datasets):
    source = _source([datasets / "truth.jsonl"])
    task = source.next(0)
    assert task.metadata["source"] == "truth.jsonl"
    assert task.metadata["record_id"].startswith("truth-")
    assert isinstance(task.metadata["line"], int)
    assert task.metadata["synthetic_label"] is True
    assert task.metadata["record_ids"]


def test_determinism_across_hash_seeds(datasets):
    """The label strategy must not use the salted builtin ``hash``."""
    script = (
        "import sys;"
        "from mactf.data.sources import JsonlTaskSource;"
        f"src = JsonlTaskSource([{str(datasets / 'truth.jsonl')!r}],"
        " state_dim=6, action_dim=4, batch_size=8, seed=11);"
        "t = src.next(0);"
        "print(t.targets.tolist(), [round(float(x), 9) for x in t.states.flatten()[:6]])"
    )
    outputs = []
    for seed in ("0", "12345", "random"):
        environment = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(ROOT))
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=environment,
            timeout=180,
            cwd=str(ROOT),
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(completed.stdout.strip())
    assert len(set(outputs)) == 1, outputs


def test_epoch_shuffle_is_deterministic(datasets):
    first = _source([datasets / "interaction.jsonl"], shuffle=True)
    second = _source([datasets / "interaction.jsonl"], shuffle=True)
    for epoch in range(3):
        assert first.next(epoch).targets.tolist() == second.next(epoch).targets.tolist()


def test_unshuffled_keeps_file_order(datasets):
    source = _source([datasets / "interaction.jsonl"], shuffle=False)
    task = source.next(0)
    assert task.metadata["record_ids"] == [
        "interaction-001",
        "interaction-002",
        "interaction-003",
        "interaction-004",
    ]


def test_short_dataset_yields_one_short_batch(datasets):
    source = _source([datasets / "truth.jsonl"], batch_size=32)
    task = source.next(0)
    assert task.states.shape[0] == 3
    assert len(source) == 1


def test_explicit_strategy_skips_text_only_records(tmp_path):
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "prompt": "text only"}),
                json.dumps({"id": "b", "states": [[0.1] * 8], "targets": [1]}),
            ]
        ),
        encoding="utf-8",
    )
    source = JsonlTaskSource(
        [path], state_dim=8, action_dim=4, batch_size=8, label_strategy="explicit"
    )
    assert [record.id for record in source.records()] == ["b"]


def test_explicit_strategy_with_no_records_raises(tmp_path):
    path = tmp_path / "text.jsonl"
    path.write_text(json.dumps({"id": "a", "prompt": "text"}), encoding="utf-8")
    with pytest.raises(DataError):
        JsonlTaskSource(
            [path], state_dim=8, action_dim=4, batch_size=8, label_strategy="explicit"
        )


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DataError):
        JsonlTaskSource([path], state_dim=8, action_dim=4, batch_size=8)


def test_duplicate_id_in_one_file_raises(tmp_path):
    path = tmp_path / "dup.jsonl"
    path.write_text('{"id": "a"}\n{"id": "a"}\n', encoding="utf-8")
    with pytest.raises(DataError) as captured:
        JsonlTaskSource([path], state_dim=8, action_dim=4, batch_size=8)
    assert "duplicate" in str(captured.value)


def test_wrong_state_width_raises(tmp_path):
    path = tmp_path / "wide.jsonl"
    path.write_text(
        json.dumps({"id": "a", "states": [[1.0, 2.0, 3.0]], "targets": [0]}),
        encoding="utf-8",
    )
    with pytest.raises(DataError) as captured:
        JsonlTaskSource([path], state_dim=8, action_dim=4, batch_size=8)
    assert "state_dim" in str(captured.value)


def test_mixed_source_is_deterministic(datasets):
    def build():
        return MixedTaskSource(
            [
                (_source([datasets / "truth.jsonl"]), 1.0),
                (_source([datasets / "ambiguity.jsonl"]), 1.0),
            ]
        )

    first, second = build(), build()
    for epoch in range(4):
        assert first.next(epoch).metadata["source"] == second.next(epoch).metadata["source"]


def test_mixed_source_rejects_bad_weights(datasets):
    with pytest.raises(ValueError):
        MixedTaskSource([(_source([datasets / "truth.jsonl"]), -1.0)])
    with pytest.raises(ValueError):
        MixedTaskSource([(_source([datasets / "truth.jsonl"]), 0.0)])


def test_synthetic_is_cross_process_stable():
    """Regression guard: the synthetic generator must stay hash-salt stable."""
    script = (
        "from mactf.data.sources import SyntheticTaskSource;"
        "src = SyntheticTaskSource(state_dim=6, action_dim=3, batch_size=4, seed=99);"
        "t = src.next(2);"
        "print(t.targets.tolist(), [round(float(x), 9) for x in t.states.flatten()[:5]])"
    )
    outputs = []
    for seed in ("0", "12345", "random"):
        environment = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(ROOT))
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=environment,
            timeout=180,
            cwd=str(ROOT),
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(completed.stdout.strip())
    assert len(set(outputs)) == 1, outputs
