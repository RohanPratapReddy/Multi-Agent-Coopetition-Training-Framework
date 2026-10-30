"""``mactf migrate`` (document 16, section 8.6)."""

from __future__ import annotations

from pathlib import Path


def database(project) -> Path:
    return project.package / "var" / "mactf.sqlite3"


def test_check_and_plan_never_create_the_database(project):
    for flag in ("--check", "--plan"):
        out = project.invoke("migrate", flag, "--format", "json")
        assert out.json()["exists"] is False and out.json()["schema_version"] == 0
        assert out.json()["pending"], "a missing database has everything pending"
    assert not any(project.root.rglob("*.sqlite3"))


def test_check_fails_when_behind_but_plan_only_reports(project):
    assert project.invoke("migrate", "--check").code == 1
    assert project.invoke("migrate", "--plan").code == 0


def test_check_and_plan_are_mutually_exclusive(project):
    assert project.invoke("migrate", "--check", "--plan").code == 2


def test_apply_creates_the_database_then_is_idempotent(project):
    first = project.invoke("migrate", "--database", str(project.root / "reg.sqlite3"), "--format", "json")
    assert first.code == 0, first.stderr
    payload = first.json()
    assert payload["up_to_date"] is True and payload["applied"] and payload["pending"] == []
    second = project.invoke("migrate", "--database", str(project.root / "reg.sqlite3"), "--format", "json")
    assert second.code == 0 and second.json()["applied"] == []
    assert "no migrations needed" in project.invoke(
        "migrate", "--database", str(project.root / "reg.sqlite3")).stdout


def test_check_succeeds_once_migrated(project):
    path = str(project.root / "reg.sqlite3")
    assert project.invoke("migrate", "--database", path).code == 0
    out = project.invoke("migrate", "--check", "--database", path, "--format", "json")
    assert out.code == 0 and out.json()["up_to_date"] is True and out.json()["exists"] is True


def test_plan_text_says_the_database_is_not_created_yet(project):
    out = project.invoke("migrate", "--plan")
    assert out.code == 0 and "not created yet" in out.stdout and "pending" in out.stdout


def test_default_database_comes_from_the_settings(project):
    out = project.invoke("migrate", "--format", "json")
    assert out.code == 0, out.stderr
    assert Path(out.json()["database"]) == database(project).resolve() or database(project).exists()
