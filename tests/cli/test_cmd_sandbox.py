"""``mactf sandbox`` (document 16, section 8.10). No test here needs QEMU installed."""

from __future__ import annotations

import sys

import pytest

from mactf.management.commands import sandbox as sandbox_command
from mactf.sandbox.base import IsolationLevel
from mactf.sandbox.detect import probe_all


def best_isolation() -> IsolationLevel:
    return max((report.isolation for report in probe_all() if report.usable), default=IsolationLevel.NONE)


def test_status_is_the_default_verb(project):
    bare = project.invoke("sandbox")
    explicit = project.invoke("sandbox", "status")
    assert bare.code == explicit.code == 0
    last = lambda text: text.strip().splitlines()[-1]                   # noqa: E731 - probe temp names differ
    assert "best usable isolation" in bare.stdout and last(bare.stdout) == last(explicit.stdout)


def test_status_json_is_the_bare_array_of_reports(project):
    out = project.invoke("sandbox", "status", "--format", "json")
    assert out.code == 0
    reports = out.json()
    assert isinstance(reports, list) and reports
    assert all({"provider", "isolation", "usable"} <= set(item) for item in reports)
    assert {item["provider"] for item in reports} >= {"qemu", "local"}


def test_require_none_always_passes_and_hardware_follows_the_machine(project):
    assert project.invoke("sandbox", "--require", "none").code == 0
    assert project.invoke("sandbox", "status", "--require", "none").code == 0
    wanted = 0 if best_isolation() >= IsolationLevel.HARDWARE else 5
    for argv in (("sandbox", "--require", "hardware"), ("sandbox", "status", "--require", "hardware"),
                 ("sandbox", "--require", "hardware", "status")):
        out = project.invoke(*argv)
        assert out.code == wanted, (argv, out.stderr)
        if wanted:
            assert "required" in out.stderr


def test_require_rejects_an_unknown_level(project):
    assert project.invoke("sandbox", "--require", "magic").code == 2


def test_install_is_a_no_op_when_qemu_is_present(project, monkeypatch):
    monkeypatch.setattr(sandbox_command.shutil, "which", lambda name: "/usr/bin/" + name)
    out = project.invoke("sandbox", "install")
    assert out.code == 0 and "already installed" in out.stdout


def test_install_dry_run_prints_the_platform_command(project, monkeypatch):
    package_manager = "winget" if sys.platform.startswith("win") else "apt-get"
    if not sys.platform.startswith(("win", "linux")):
        pytest.skip("no automatic installer on this platform")
    monkeypatch.setattr(sandbox_command.shutil, "which",
                        lambda name: f"/bin/{name}" if name == package_manager else None)
    monkeypatch.setattr(sandbox_command.os, "geteuid", lambda: 0, raising=False)
    out = project.invoke("sandbox", "install", "--dry-run")
    assert out.code == 0 and "would run:" in out.stdout and "qemu" in out.stdout.lower()


def test_install_without_a_package_manager_is_unavailable(project, monkeypatch):
    monkeypatch.setattr(sandbox_command.shutil, "which", lambda name: None)
    out = project.invoke("sandbox", "install", "--dry-run")
    assert out.code == 5 and out.stderr


def test_install_will_not_prompt_without_a_terminal(project, monkeypatch):
    if not sys.platform.startswith(("win", "linux")):
        pytest.skip("no automatic installer on this platform")
    monkeypatch.setattr(sandbox_command.shutil, "which",
                        lambda name: f"/bin/{name}" if name in ("winget", "apt-get", "dnf", "pacman") else None)
    monkeypatch.setattr(sandbox_command.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(sandbox_command.subprocess, "run",
                        lambda *a, **k: pytest.fail("the installer must not run unconfirmed"))
    assert project.invoke("sandbox", "install").code == 2


def test_provision_needs_a_base_image_and_says_where_to_get_one(project):
    out = project.invoke("sandbox", "provision", "--peer", "alpha", "--isolation", "none")
    assert out.code in (1, 5) and out.stderr
    if out.code == 1:
        assert "base image missing" in out.stderr + out.stdout


def test_no_fallback_refuses_a_weaker_provider(project):
    if best_isolation() >= IsolationLevel.HARDWARE:
        pytest.skip("hardware isolation is available here, so there is nothing to refuse")
    out = project.invoke("sandbox", "provision", "--isolation", "hardware", "--no-fallback")
    assert out.code == 5


def test_provision_rejects_an_unknown_isolation(project):
    assert project.invoke("sandbox", "provision", "--isolation", "magic").code == 2
    assert project.invoke("sandbox", "launch", "--isolation", "magic").code == 2


def sandbox_files(project):
    images = project.package / "var" / "images"
    run = project.package / "var" / "run"
    images.mkdir(parents=True, exist_ok=True)
    run.mkdir(parents=True, exist_ok=True)
    base = images / "base.qcow2"
    base.write_bytes(b"base image")
    work = [images / "peer-alpha.qcow2", images / "peer-beta.qcow2"]
    sockets = [run / "mactf-alpha.sock"]
    for path in (*work, *sockets):
        path.write_bytes(b"x")
    return base, work, sockets


def test_clean_dry_run_lists_and_removes_nothing(project):
    base, work, sockets = sandbox_files(project)
    out = project.invoke("sandbox", "clean", "--dry-run", "--format", "json")
    assert out.code == 0 and out.json()["count"] == 3 and out.json()["dry_run"] is True
    assert all(path.exists() for path in (base, *work, *sockets))


def test_clean_removes_overlays_and_sockets_but_never_the_base_image(project):
    base, work, sockets = sandbox_files(project)
    out = project.invoke("sandbox", "clean", "--yes", "--format", "json")
    assert out.code == 0, out.stderr
    assert out.json()["count"] == 3 and out.json()["failed"] == []
    assert base.exists() and not any(path.exists() for path in (*work, *sockets))
    assert project.invoke("sandbox", "clean", "--yes", "--format", "json").json()["count"] == 0


def test_clean_will_not_prompt_without_a_terminal(project):
    base, work, sockets = sandbox_files(project)
    assert project.invoke("sandbox", "clean").code == 2
    assert all(path.exists() for path in (*work, *sockets))


def test_clean_with_nothing_to_clean_needs_no_confirmation(project):
    out = project.invoke("sandbox", "clean")
    assert out.code == 0 and "0 sandbox artifact" in out.stdout


def test_clean_reports_a_file_it_cannot_remove(project, monkeypatch):
    base, work, sockets = sandbox_files(project)
    from pathlib import Path

    real_unlink = Path.unlink

    def refuse(self, *args, **kwargs):
        if self.name == "peer-alpha.qcow2":
            raise PermissionError(13, "Permission denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse)
    out = project.invoke("sandbox", "clean", "--yes", "--format", "json")
    assert out.code == 1 and [item["path"].endswith("peer-alpha.qcow2") for item in out.json()["failed"]] == [True]
    assert work[0].exists() and not work[1].exists()
