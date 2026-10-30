"""``mactf doctor`` (document 16, section 8.8)."""

from __future__ import annotations

from mactf.testing.cli import invoke


def test_doctor_reports_without_any_project():
    out = invoke(["doctor", "--format", "json"])
    assert out.code == 0, out.stderr
    payload = out.json()
    assert payload["python"] and payload["mactf"] and payload["isolation_available"]
    assert {"torch", "cuda_available", "qemu", "sandbox", "gpus", "project", "requirements"} <= set(payload)
    assert payload["project"] is None and payload["settings_module"] == "mactf.settings_default"


def test_doctor_text_is_a_readable_report():
    out = invoke(["doctor"])
    assert out.code == 0 and out.stdout.startswith("MACTF doctor") and "isolation" in out.stdout


def test_doctor_in_a_project_names_the_settings_module(project):
    out = project.invoke("doctor", "--format", "json")
    assert out.code == 0 and out.json()["settings_module"] == project.module


def test_doctor_survives_settings_that_do_not_load(project):
    (project.package / "settings.py").write_text("raise RuntimeError('broken')\n", encoding="utf-8")
    out = project.invoke("doctor", "--format", "json")
    assert out.code == 1
    assert "broken" in out.json()["settings_error"] and out.json()["ok"] is False


def test_require_a_met_capability_passes_and_an_unmet_one_exits_5(project):
    base = project.invoke("doctor", "--format", "json").json()
    for name, met in (("cuda", base["cuda_available"]), ("qemu", bool(base["qemu"]))):
        out = project.invoke("doctor", "--require", name, "--format", "json")
        assert out.code == (0 if met else 5), (name, out.stdout, out.stderr)
        assert out.json()["requirements"] == [{"name": name, "met": met}]


def test_require_rejects_an_unknown_capability():
    assert invoke(["doctor", "--require", "teleport"]).code == 2
