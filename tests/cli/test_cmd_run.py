"""``mactf run`` (document 16, section 8.2): real runs of the generated three-peer swarm on the CPU."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

CPU = ("--device", "cpu")


def run_json(project, *argv):
    out = project.invoke("run", *argv, "--format", "json", *CPU)
    return out, (out.json() if out.stdout.strip().startswith("{") else None)


def registry_of(project):
    from mactf import conf
    from mactf.db import Registry

    return Registry.from_settings(conf.load_settings(project.module))


def log_of(project, run_id):
    return project.package / "var" / "runs" / run_id / "events.jsonl"


def test_a_run_completes_records_itself_and_writes_a_log(project):
    out, payload = run_json(project, "--epochs", "2", "--run-id", "first")
    assert out.code == 0, out.stderr
    assert payload["run_id"] == "first" and payload["status"] == "completed"
    assert payload["epochs_completed"] == 2 and payload["mode"] == "new" and payload["exit_code"] == 0
    row = registry_of(project).get_run("first")
    assert row["status"] == "completed"
    assert log_of(project, "first").exists()
    assert str(log_of(project, "first")) == payload["log_file"]


def test_jsonl_streams_start_epochs_and_end(project):
    out = project.invoke("run", "--epochs", "2", "--run-id", "stream", "--format", "jsonl", *CPU)
    assert out.code == 0, out.stderr
    events = [item["event"] for item in out.jsonl()]
    assert events[0] == "start" and events[-1] == "end" and events.count("epoch") == 2
    epoch = next(item for item in out.jsonl() if item["event"] == "epoch")
    assert {"epoch", "epochs", "score", "kappa", "entropy", "weights"} <= set(epoch)


def test_text_output_prints_a_line_per_epoch(project):
    out = project.invoke("run", "--epochs", "2", *CPU)
    assert out.code == 0, out.stderr
    assert "completed" in out.stdout


def test_dry_run_records_a_dry_run_row_and_trains_nothing(project):
    out, payload = run_json(project, "--dry-run", "--run-id", "dry")
    assert out.code == 0 and payload["status"] == "dry_run"
    assert registry_of(project).get_run("dry")["status"] == "dry_run"
    assert not list((project.package / "var" / "checkpoints").glob("dry-*"))


def test_a_run_id_is_never_reused(project):
    assert run_json(project, "--epochs", "1", "--run-id", "once")[0].code == 0
    again = project.invoke("run", "--epochs", "1", "--run-id", "once", *CPU)
    assert again.code == 4 and "already exists" in again.stderr


def test_bad_arguments_are_usage_errors(project):
    assert project.invoke("run", "--epochs", "0").code == 2
    assert project.invoke("run", "--run-id", "bad id!").code == 2
    assert project.invoke("run", "--timeout", "soon").code == 2
    assert project.invoke("run", "--device", "tpu").code == 2
    assert project.invoke("run", "--redteam", "--no-redteam").code == 2
    assert project.invoke("run", "--log-file", "x", "--no-log").code == 2
    assert project.invoke("run", "--resume", "--dry-run").code == 2


def test_no_log_writes_no_per_run_log(project):
    out, payload = run_json(project, "--epochs", "1", "--run-id", "quiet", "--no-log")
    assert out.code == 0 and not log_of(project, "quiet").exists() and payload["log_file"] is None


def test_log_file_redirects_the_per_run_log(project, tmp_path):
    target = tmp_path / "elsewhere" / "mine.jsonl"
    out, payload = run_json(project, "--epochs", "1", "--run-id", "moved", "--log-file", str(target))
    assert out.code == 0 and target.exists() and not log_of(project, "moved").exists()
    records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line]
    assert records and all(record["run_id"] == "moved" for record in records if record.get("run_id"))


def test_set_overrides_a_setting_for_this_run(project):
    out, payload = run_json(project, "--run-id", "tuned", "--set", "TRAINING.EPOCHS=2")
    assert out.code == 0, out.stderr
    assert payload["epochs"] == 2 and payload["epochs_completed"] == 2
    bad = project.invoke("run", "--set", "NOT.A.REAL.SETTING=1", *CPU)
    assert bad.code != 0 or "NOT" in bad.stderr


def test_a_timeout_pauses_the_run_and_resume_finishes_it(project):
    out, payload = run_json(project, "--epochs", "6", "--run-id", "slow", "--timeout", "0.01")
    assert out.code == 124 and payload["status"] == "paused" and payload["stopped_by"] == "timeout"
    assert registry_of(project).get_run("slow")["status"] == "paused"

    resumed, again = run_json(project, "--resume", "--epochs", "3")
    assert resumed.code == 0, resumed.stderr
    assert again["run_id"] == "slow" and again["mode"] == "resume" and again["status"] == "completed"
    assert registry_of(project).get_run("slow")["status"] == "completed"


def test_resume_with_nothing_resumable_is_not_found(project):
    assert project.invoke("run", "--resume", *CPU).code == 3
    assert run_json(project, "--epochs", "1", "--run-id", "done")[0].code == 0
    assert project.invoke("run", "--resume", "done", *CPU).code == 4       # completed runs do not resume


def test_branch_from_starts_a_new_run_from_a_checkpoint(project):
    assert run_json(project, "--epochs", "3", "--run-id", "trunk")[0].code == 0
    out, payload = run_json(project, "--branch-from", "trunk:1", "--epochs", "3", "--run-id", "twig")
    assert out.code == 0, out.stderr
    assert payload["mode"] == "branch" and payload["source_run"] == "trunk" and payload["source_episode"] == 1
    assert registry_of(project).get_run("twig")["status"] == "completed"
    assert registry_of(project).get_run("trunk")["status"] == "completed"


def test_branch_from_an_unknown_run_is_not_found(project):
    assert project.invoke("run", "--branch-from", "ghost:1", "--epochs", "2", *CPU).code == 3


def test_a_finished_run_is_visible_to_runs_and_logs(project):
    assert run_json(project, "--epochs", "2", "--run-id", "seen")[0].code == 0
    listing = project.invoke("runs", "list", "--format", "json").json()
    assert [item["id"] for item in listing["items"]] == ["seen"]
    assert listing["items"][0]["liveness"] == "idle"
    tail = project.invoke("logs", "tail", "seen", "--format", "jsonl")
    assert tail.code == 0 and tail.jsonl()
    assert project.invoke("checkpoints", "verify", "seen").code == 0
