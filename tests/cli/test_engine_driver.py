"""``mactf run`` / ``runs resume`` / ``redteam`` select the engine from configuration (document 17 section 13.6, rows 23, 24, 29)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

CPU = ("--device", "cpu")
TRAINER = ("--set", "ORCHESTRATOR.ENGINE=trainer")


def registry_of(project):
    from mactf import conf
    from mactf.db import Registry

    return Registry.from_settings(conf.load_settings(project.module))


def test_cli_run_selects_the_engine(project):
    out = project.invoke("run", "--epochs", "2", "--run-id", "t2", "--format", "jsonl", *TRAINER, *CPU)
    assert out.code == 0, out.stdout + out.stderr
    events = [item["event"] for item in out.jsonl()]
    assert events[0] == "start" and events[-1] == "end" and events.count("epoch") == 2
    row = registry_of(project).get_run("t2")
    assert row["status"] == "completed" and json.loads(row["settings"])["engine"] == "trainer"
    # the default is untouched: no key, no ``engine`` in the row
    plain = project.invoke("run", "--epochs", "1", "--run-id", "o1", "--format", "json", *CPU)
    assert plain.code == 0, plain.stderr
    assert "engine" not in json.loads(registry_of(project).get_run("o1")["settings"])


def test_a_bad_engine_fails_before_a_model_exists(project):
    out = project.invoke("run", "--epochs", "1", "--set", "ORCHESTRATOR.ENGINE=xyz", *CPU)
    assert out.code != 0 and "E090" in (out.stdout + out.stderr)


def test_cli_resume_and_branch(project):
    first = project.invoke("run", "--epochs", "2", "--run-id", "r1", "--format", "json", *TRAINER, *CPU)
    assert first.code == 0, first.stderr
    registry_of(project).set_run_status("r1", "paused", finished=False)
    resumed = project.invoke("runs", "resume", "r1", "--epochs", "4", "--format", "json", *TRAINER, *CPU)
    assert resumed.code == 0, resumed.stdout + resumed.stderr
    assert resumed.json()["status"] == "completed" and resumed.json()["epochs_completed"] == 2
    assert registry_of(project).count_episodes("r1") == 4
    branch = project.invoke("run", "--epochs", "5", "--run-id", "b1", "--branch-from", "r1", "--format", "json", *TRAINER, *CPU)
    assert branch.code == 0, branch.stdout + branch.stderr
    assert branch.json()["status"] == "completed" and branch.json()["source_run"] == "r1"
