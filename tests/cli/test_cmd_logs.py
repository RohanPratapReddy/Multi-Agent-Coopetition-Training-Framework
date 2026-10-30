"""``mactf logs`` (document 16, section 8.14)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mactf.management.commands.logs import render_log_line


def record(kind="epoch", level="info", run_id="run-completed-a", node="trainer", epoch=1, message="", **fields):
    return {"kind": kind, "level": level, "run_id": run_id, "node_path": node, "epoch": epoch,
            "wall_time": 1_760_000_000.0 + epoch, "message": message, "fields": fields}


def write_log(path: Path, records, *, raw_tail: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item) + "\n" for item in records) + raw_tail, encoding="utf-8")
    return path


def own_log(seeded, run_id="run-completed-a") -> Path:
    return seeded.project.root / "demo_swarm" / "var" / "runs" / run_id / "events.jsonl"


def test_render_log_line_survives_odd_records():
    assert "epoch" in render_log_line(record())
    assert "INFO" in render_log_line({})                  # every missing key is defaulted
    line = render_log_line({"kind": 7, "depth": "x", "wall_time": "never", "fields": 3, "extra": True})
    assert "value=3" in line and "7" in line


def test_tail_prints_the_last_n_records_oldest_first(project, tmp_path):
    path = write_log(tmp_path / "x.jsonl", [record(epoch=n, message=f"m{n}") for n in range(1, 8)])
    out = project.invoke("logs", "tail", "--file", str(path), "-n", "3")
    assert out.code == 0
    lines = out.stdout.splitlines()
    assert len(lines) == 3 and "m5" in lines[0] and "m7" in lines[2]


def test_tail_filters_by_kind_level_and_node(project, tmp_path):
    path = write_log(tmp_path / "x.jsonl", [
        record("epoch", "info", node="trainer", message="a"),
        record("warning", "warning", node="peer.a", message="b"),
        record("loss", "error", node="peer.a", message="c"),
        record("loss", "info", node="peer.b", message="d"),
    ])

    def messages(*argv):
        out = project.invoke("logs", "tail", "--file", str(path), "--format", "jsonl", *argv)
        assert out.code == 0, out.stderr
        return [item["message"] for item in out.jsonl()]

    assert messages("--kind", "loss") == ["c", "d"]
    assert messages("--level", "warning") == ["b"]
    assert messages("--node", "peer.a") == ["b", "c"]
    assert messages("--kind", "loss", "--node", "peer.b") == ["d"]


def test_tail_json_is_one_document(project, tmp_path):
    path = write_log(tmp_path / "x.jsonl", [record(message="one"), record(message="two")])
    out = project.invoke("logs", "tail", "--file", str(path), "--format", "json")
    assert out.code == 0
    payload = out.json()
    assert payload["count"] == 2 and [i["message"] for i in payload["items"]] == ["one", "two"]


def test_tail_ignores_malformed_and_partial_lines(project, tmp_path):
    path = write_log(tmp_path / "x.jsonl", [record(message="good")], raw_tail='not json\n{"kind": "epo')
    out = project.invoke("logs", "tail", "--file", str(path), "--format", "jsonl")
    assert out.code == 0 and [item["message"] for item in out.jsonl()] == ["good"]


def test_tail_of_a_missing_file_is_not_found(project, tmp_path):
    assert project.invoke("logs", "tail", "--file", str(tmp_path / "missing.jsonl")).code == 3


def test_tail_rejects_bad_options(project, tmp_path):
    path = write_log(tmp_path / "x.jsonl", [record()])
    assert project.invoke("logs", "tail", "--file", str(path), "-n", "0").code == 2
    assert project.invoke("logs", "tail", "--file", str(path), "--level", "loud").code == 2
    assert project.invoke("logs", "tail", "--file", str(path), "-f", "--format", "json").code == 2


def test_tail_reads_a_runs_own_log(seeded):
    write_log(own_log(seeded), [record(message="hello from the run")])
    out = seeded.invoke("logs", "tail", "run-completed-a")
    assert out.code == 0 and "hello from the run" in out.stdout


def test_tail_a_run_without_a_log_says_where_it_looked(seeded):
    out = seeded.invoke("logs", "tail", "run-completed-a")
    assert out.code == 3 and "events.jsonl" in out.stderr


def test_path_reports_where_the_log_is_or_would_be(seeded):
    before = seeded.invoke("logs", "path", "run-completed-a", "--format", "json").json()
    assert before["exists"] is False and before["path"].endswith("events.jsonl")
    assert not own_log(seeded).exists()                         # asking creates nothing
    write_log(own_log(seeded), [record()])
    after = seeded.invoke("logs", "path", "run-completed-a", "--format", "json").json()
    assert after["exists"] is True and after["path"] == before["path"]
    assert seeded.invoke("logs", "path", "run-completed-a").stdout.strip() == after["path"]


def test_path_for_an_unknown_run_is_not_found(seeded):
    assert seeded.invoke("logs", "path", "nope").code == 3


def test_follow_on_a_finished_run_prints_the_tail_and_returns(seeded):
    write_log(own_log(seeded), [record(message="before")])
    out = seeded.invoke("logs", "tail", "run-completed-a", "-f")
    assert out.code == 0 and "before" in out.stdout


@pytest.mark.parametrize("fmt", ["text", "jsonl"])
def test_follow_a_stale_run_ends_on_its_own(seeded, fmt):
    write_log(own_log(seeded, "run-running-dead"), [record(run_id="run-running-dead", message="last words")])
    out = seeded.invoke("logs", "tail", "run-running-dead", "-f", "--format", fmt)
    assert out.code == 0 and "last words" in out.stdout
