import hashlib
import os
from pathlib import Path

import pytest


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts or path.name.endswith(("-wal", "-shm")):
            continue                        # bytecode, and SQLite's own reader side files
        digest.update(str(path.relative_to(root)).encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def listing(seeded, *argv):
    out = seeded.invoke("runs", "list", *argv, "--format", "json")
    assert out.code == 0, out.stderr
    return [item["id"] for item in out.json()["items"]]


def test_list_newest_first_and_reverse(seeded):
    ids = listing(seeded, "--all")
    assert ids[0] == "run-null-metrics" and ids[-1] == "run-completed-a" and len(ids) == 9
    assert listing(seeded, "--all", "--reverse") == ids[::-1]


def test_list_status_groups(seeded):
    assert set(listing(seeded, "--status", "active")) == {"run-running-dead", "run-running-live"}
    assert set(listing(seeded, "--status", "resumable")) == {"run-paused", "run-interrupted"}
    final = set(listing(seeded, "--status", "final"))
    assert {"run-completed-a", "run-failed", "run-cancelled"} <= final and "run-paused" not in final
    assert set(listing(seeded, "--status", "failed", "--status", "paused")) == {"run-failed", "run-paused"}


def test_list_prefix_limit_and_exclusivity(seeded):
    assert set(listing(seeded, "--prefix", "run-completed")) == {"run-completed-a", "run-completed-b"}
    assert len(listing(seeded, "--limit", "2")) == 2
    assert seeded.invoke("runs", "list", "--limit", "2", "--all").code == 2


def test_list_time_filters(seeded):
    assert len(listing(seeded, "--since", "1h", "--all")) == 9
    assert listing(seeded, "--until", "1h", "--all") == []
    assert seeded.invoke("runs", "list", "--since", "bogus").code == 2
    assert seeded.invoke("runs", "list", "--since", "1h", "--until", "2h").code == 2


def test_list_jsonl_has_one_record_per_run(seeded):
    out = seeded.invoke("runs", "list", "--all", "--format", "jsonl")
    assert out.code == 0 and [item["id"] for item in out.jsonl()] == listing(seeded, "--all")


def test_list_marks_a_stale_run_and_a_live_one(seeded):
    items = {i["id"]: i for i in seeded.invoke("runs", "list", "--all", "--format", "json").json()["items"]}
    assert items["run-running-live"]["liveness"] == "live"
    assert items["run-running-dead"]["liveness"] == "stale"


def test_empty_registry_and_missing_database_succeed(project):
    out = project.invoke("runs", "list", "--format", "json")
    assert out.code == 0 and out.json()["items"] == [] and out.json()["total"] == 0
    assert project.invoke("runs", "list").code == 0


def test_read_only_verbs_create_nothing(project):
    before = tree_hash(project.root)
    for verb in (("list",), ("show",), ("show", "latest")):
        project.invoke("runs", *verb)
    assert tree_hash(project.root) == before                  # no database, lock, or log appeared


def test_show_and_latest_and_unique_prefix(seeded):
    out = seeded.invoke("runs", "show", "--format", "json")
    assert out.code == 0 and out.json()["run"]["id"] == "run-null-metrics"
    assert seeded.invoke("runs", "show", "run-paused", "--format", "json").json()["run"]["status"] == "paused"
    assert seeded.invoke("runs", "show", "run-pau", "--format", "json").json()["run"]["id"] == "run-paused"


def test_ambiguous_prefix_is_usage_and_unknown_is_not_found(seeded):
    out = seeded.invoke("runs", "show", "run-comp")
    assert out.code == 2 and "run-completed-a" in out.stderr
    assert seeded.invoke("runs", "show", "nope").code == 3


def test_show_sections(seeded):
    plain = seeded.invoke("runs", "show", "run-completed-a", "--format", "json").json()
    full = seeded.invoke("runs", "show", "run-completed-a", "--all-sections", "--format", "json").json()
    assert set(plain) < set(full)


def test_wait_returns_for_a_finished_run_and_times_out_on_a_live_one(seeded):
    assert seeded.invoke("runs", "wait", "run-completed-a").code == 0
    assert seeded.invoke("runs", "wait", "run-running-dead").code == 1          # stale: stopped, but not completed
    out = seeded.invoke("runs", "wait", "run-running-live", "--timeout", "1s", "--interval", "1s")
    assert out.code == 124


def test_cancel_refuses_a_live_run_and_cancels_a_dead_one(seeded):
    assert seeded.invoke("runs", "cancel", "run-running-live", "--yes").code == 4
    assert seeded.invoke("runs", "cancel", "run-running-dead", "--dry-run").code == 0
    assert seeded.registry.get_run("run-running-dead")["status"] == "running"
    assert seeded.invoke("runs", "cancel", "run-running-dead", "--yes", "--reason", "bad seed").code == 0
    row = seeded.registry.get_run("run-running-dead")
    assert row["status"] == "cancelled" and "bad seed" in (row["notes"] or "")


def test_destructive_verbs_will_not_prompt_without_a_terminal(seeded):
    out = seeded.invoke("runs", "cancel", "run-running-dead")
    assert out.code == 2 and seeded.registry.get_run("run-running-dead")["status"] == "running"


def test_reap_marks_only_the_dead_run(seeded):
    assert seeded.invoke("runs", "reap", "--dry-run").code == 0
    assert seeded.registry.get_run("run-running-dead")["status"] == "running"
    out = seeded.invoke("runs", "reap", "--yes", "--format", "json")
    assert out.code == 0, out.stderr
    assert seeded.registry.get_run("run-running-dead")["status"] == "interrupted"
    assert seeded.registry.get_run("run-running-live")["status"] == "running"


def test_resume_refuses_a_completed_run(seeded):
    assert seeded.invoke("runs", "resume", "run-completed-a").code == 4
    assert seeded.invoke("runs", "resume", "run-running-live").code == 4
    assert seeded.invoke("runs", "resume", "nope").code == 3


def test_delete_dry_run_changes_nothing_and_yes_removes_rows_and_files(seeded):
    before = tree_hash(seeded.project.root)
    assert seeded.invoke("runs", "delete", "run-completed-a", "--dry-run").code == 0
    assert tree_hash(seeded.project.root) == before and seeded.registry.get_run("run-completed-a")
    files = sorted(seeded.checkpoint_dir.glob("run-completed-a-*.pt"))
    assert len(files) == 3
    out = seeded.invoke("runs", "delete", "run-completed-a", "--yes")
    assert out.code == 0, out.stderr
    assert seeded.registry.get_run("run-completed-a") is None
    assert not any(path.exists() for path in files)
    assert seeded.registry.get_run("run-completed-b") is not None


def test_delete_keep_files_leaves_checkpoints(seeded):
    assert seeded.invoke("runs", "rm", "run-paused", "--yes", "--keep-files").code == 0
    assert seeded.registry.get_run("run-paused") is None
    assert len(list(seeded.checkpoint_dir.glob("run-paused-*.pt"))) == 2


def test_delete_refuses_a_live_run(seeded):
    assert seeded.invoke("runs", "delete", "run-running-live", "--yes").code == 4
    assert seeded.registry.get_run("run-running-live") is not None


def test_delete_older_than_selects_nothing_for_fresh_runs(seeded):
    out = seeded.invoke("runs", "delete", "--older-than", "30d", "--yes", "--format", "json")
    assert out.code == 0 and len(listing(seeded, "--all")) == 9
