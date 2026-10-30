"""``mactf redteam`` (document 16, section 8.3)."""

from __future__ import annotations

import pytest

from mactf.management.commands.redteam import INJECTORS

pytest.importorskip("torch")

CPU = ("--device", "cpu")


def test_list_injectors_trains_nothing(project):
    out = project.invoke("redteam", "--list-injectors", "--format", "json")
    assert out.code == 0
    assert [item["name"] for item in out.json()["injectors"]] == sorted(INJECTORS)
    assert all(item["description"] for item in out.json()["injectors"])
    assert "INJECTOR" in project.invoke("redteam", "--list-injectors").stdout
    assert not any(project.root.rglob("*.sqlite3"))


def test_every_injector_passes_against_a_healthy_swarm(project):
    out = project.invoke("redteam", "--epochs", "1", "--run-id", "rt", "--format", "json", *CPU)
    assert out.code == 0, out.stdout + out.stderr
    payload = out.json()
    assert payload["injectors"] == len(INJECTORS) and payload["failed"] == 0
    assert {item["injector"] for item in payload["findings"]} == set(INJECTORS)
    assert all(item["status"] == "ok" for item in payload["findings"])


def test_the_scores_show_p20_voiding_the_episode(project):
    out = project.invoke("redteam", "--epochs", "1", "--injector", "jailbreak_attempt", "--format", "json", *CPU)
    assert out.code == 0, out.stderr
    card = out.json()["scores"]["jailbreak_attempt"]
    assert card["void"] is True and card["final_score"] < 0 and "P20" in card["non_offsettable"]


def test_a_single_injector_can_be_chosen_and_repeated(project):
    one = project.invoke("redteam", "--epochs", "1", "--injector", "rogue_credit_thief", "--format", "json", *CPU)
    assert [item["injector"] for item in one.json()["findings"]] == ["rogue_credit_thief"]
    both = project.invoke("redteam", "--epochs", "1", "--injector", "rogue_credit_thief",
                          "--injector", "jailbreak_attempt", "--format", "json", *CPU)
    assert both.json()["injectors"] == 2


def test_an_unknown_injector_is_a_usage_error(project):
    assert project.invoke("redteam", "--injector", "nope").code == 2


def test_dry_run_runs_the_injectors_against_the_untrained_swarm(project):
    out = project.invoke("redteam", "--dry-run", "--format", "json", *CPU)
    assert out.code == 0, out.stderr
    assert out.json()["status"] == "dry_run" and out.json()["failed"] == 0


def test_jsonl_emits_findings_then_the_end_record(project):
    out = project.invoke("redteam", "--epochs", "1", "--format", "jsonl", *CPU)
    assert out.code == 0, out.stderr
    events = [item["event"] for item in out.jsonl()]
    assert events.count("finding") == len(INJECTORS) and events[-1] == "end"


def test_text_report_names_every_injector(project):
    out = project.invoke("redteam", "--epochs", "1", *CPU)
    assert out.code == 0 and all(name in out.stdout for name in INJECTORS)


def test_a_failing_injector_is_a_finding_and_exits_1(project, monkeypatch):
    def broken(context):
        raise AssertionError("the containment held when it should not have")

    monkeypatch.setitem(INJECTORS, "jailbreak_attempt", broken)
    out = project.invoke("redteam", "--epochs", "1", "--injector", "jailbreak_attempt", "--format", "json", *CPU)
    assert out.code == 1
    finding = out.json()["findings"][0]
    assert finding["status"] != "ok" and out.json()["failed"] == 1


def test_a_redteam_run_is_recorded_like_any_other(project):
    out = project.invoke("redteam", "--epochs", "1", "--run-id", "rt-rec", "--format", "json", *CPU)
    assert out.code == 0, out.stderr
    listing = project.invoke("runs", "list", "--format", "json").json()
    assert [item["id"] for item in listing["items"]] == ["rt-rec"]
