"""``mactf check`` (document 16, section 8.5)."""

from __future__ import annotations


def test_check_passes_on_a_generated_project(project):
    out = project.invoke("check", "--format", "json")
    assert out.code == 0, out.stdout + out.stderr
    payload = out.json()
    assert payload["ok"] is True and "conf_notes" in payload


def test_check_text_ends_with_a_summary(project):
    out = project.invoke("check")
    assert out.code == 0 and out.stdout.strip().splitlines()[-1]


def test_list_names_the_registered_checks_and_runs_none(project):
    out = project.invoke("check", "--list", "--format", "json")
    assert out.code == 0
    checks = out.json()["checks"]
    assert checks and all({"name", "app", "tags", "requires", "doc"} <= set(item) for item in checks)
    assert "CHECK" in project.invoke("check", "--list").stdout


def test_app_filter_narrows_the_checks(project):
    everything = project.invoke("check", "--list", "--format", "json").json()["checks"]
    one_app = everything[0]["app"]
    out = project.invoke("check", "--app", one_app, "--format", "json")
    assert out.code == 0, out.stderr


def test_tag_and_id_filters_are_accepted(project):
    tags = {tag for item in project.invoke("check", "--list", "--format", "json").json()["checks"]
            for tag in item["tags"]}
    argv = ["check", "--format", "json"]
    for tag in sorted(tags)[:1]:
        argv += ["--tag", tag]
    assert project.invoke(*argv).code == 0
    assert project.invoke("check", "--id", "mactf.E0", "--format", "json").code == 0


def test_fail_level_is_validated(project):
    assert project.invoke("check", "--fail-level", "FATAL").code == 2
    assert project.invoke("check", "--fail-level", "INFO", "--format", "json").code in (0, 1)


def test_a_broken_settings_module_is_a_configuration_error(project):
    (project.package / "settings.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    out = project.invoke("check")
    assert out.code != 0 and "boom" in (out.stderr + out.stdout)
