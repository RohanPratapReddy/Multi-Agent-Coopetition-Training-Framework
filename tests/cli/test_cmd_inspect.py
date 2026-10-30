"""``mactf inspect`` (document 16, section 8.7): the compatibility front for ``runs``."""

from __future__ import annotations


def test_an_empty_registry_says_so_and_succeeds(project):
    out = project.invoke("inspect", "--format", "json")
    assert out.code == 0 and out.json() == {"run": None}
    assert "no runs recorded" in project.invoke("inspect").stdout
    assert not any(project.root.rglob("*.sqlite3"))                 # looking creates nothing


def test_naming_a_run_in_an_empty_registry_is_not_found(project):
    assert project.invoke("inspect", "anything").code == 3


def test_runs_prints_the_bare_array_of_registry_rows(seeded):
    out = seeded.invoke("inspect", "--runs", "--format", "json")
    assert out.code == 0
    rows = out.json()
    assert isinstance(rows, list) and len(rows) == 9
    assert {row["id"] for row in rows} >= {"run-completed-a", "run-paused", "run-failed"}


def test_runs_table_in_text(seeded):
    out = seeded.invoke("inspect", "--runs")
    assert out.code == 0 and "run-completed-a" in out.stdout and "STATUS" in out.stdout


def test_inspect_defaults_to_the_latest_run(seeded):
    out = seeded.invoke("inspect", "--format", "json")
    assert out.code == 0 and out.json()["run"]["id"] == "run-null-metrics"


def test_inspect_agrees_with_runs_show(seeded):
    for run_id in ("run-completed-a", "run-paused", "run-failed"):
        mine = seeded.invoke("inspect", run_id, "--format", "json").json()
        theirs = seeded.invoke("runs", "show", run_id, "--format", "json").json()
        assert mine == theirs


def test_sections_are_opt_in(seeded):
    plain = seeded.invoke("inspect", "run-completed-a", "--format", "json").json()
    detailed = seeded.invoke("inspect", "run-completed-a", "--episodes", "--checkpoints",
                             "--format", "json").json()
    assert "episodes" not in plain and "checkpoints" not in plain
    assert len(detailed["episodes"]) == 4 and len(detailed["checkpoints"]) == 3


def test_prefixes_resolve_and_ambiguity_is_a_usage_error(seeded):
    assert seeded.invoke("inspect", "run-pau", "--format", "json").json()["run"]["id"] == "run-paused"
    assert seeded.invoke("inspect", "run-comp").code == 2
    assert seeded.invoke("inspect", "nope").code == 3
