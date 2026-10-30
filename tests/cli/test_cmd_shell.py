"""``mactf shell`` (document 16, section 8.9): the scripted forms; the interactive ones need a tty."""

from __future__ import annotations

import io


def test_c_runs_code_with_settings_and_apps_loaded(project):
    out = project.invoke("shell", "-c", "print(settings.module)")
    assert out.code == 0, out.stderr
    assert out.stdout.strip() == project.module


def test_the_namespace_offers_the_documented_names(project):
    names = ("settings apps rules scoring sandbox db orchestration aggregation adapter peer "
             "temperature get_settings get_registry")
    out = project.invoke("shell", "-c", f"print(all(name in globals() for name in {names.split()!r}))")
    assert out.code == 0 and out.stdout.strip() == "True"


def test_code_can_read_a_setting(project):
    out = project.invoke("shell", "-c", "print(settings['TRAINING']['EPOCHS'])")
    assert out.code == 0 and out.stdout.strip().isdigit()


def test_get_registry_creates_and_migrates_on_demand(project):
    assert not any(project.root.rglob("*.sqlite3"))
    out = project.invoke("shell", "-c", "print(get_registry().count_runs())")
    assert out.code == 0, out.stderr
    assert out.stdout.strip() == "0" and any(project.root.rglob("*.sqlite3"))


def test_a_script_is_read_from_stdin_when_there_is_no_terminal(project):
    out = project.invoke("shell", input="print(sum(range(5)))\n")
    assert out.code == 0 and out.stdout.strip() == "10"


def test_an_exception_prints_a_traceback_and_exits_1(project):
    out = project.invoke("shell", "-c", "raise ValueError('boom')")
    assert out.code == 1 and "ValueError: boom" in out.stderr


def test_a_syntax_error_exits_1(project):
    out = project.invoke("shell", "-c", "def (:")
    assert out.code == 1 and "SyntaxError" in out.stderr


def test_system_exit_codes_pass_through(project):
    assert project.invoke("shell", "-c", "raise SystemExit(7)").code == 7
    assert project.invoke("shell", "-c", "raise SystemExit").code == 0
    assert project.invoke("shell", "-c", "raise SystemExit('bye')").code == 1


def test_a_forced_missing_interpreter_is_unavailable(project, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "IPython", None)
    monkeypatch.setitem(sys.modules, "bpython", None)
    from mactf.management.commands.shell import Command

    monkeypatch.setattr(Command, "_interactive_stdin", lambda self: True)
    assert project.invoke("shell", "--interface", "ipython").code == 5
    assert project.invoke("shell", "--interface", "bpython").code == 5


def test_interface_is_validated(project):
    assert project.invoke("shell", "--interface", "emacs").code == 2


def test_a_project_whose_settings_do_not_load_fails_clearly(project):
    (project.package / "settings.py").write_text("raise RuntimeError('broken')\n", encoding="utf-8")
    out = project.invoke("shell", "-c", "print(1)")
    assert out.code != 0 and "broken" in out.stderr
