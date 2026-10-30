"""version, help, and completion (document 16, sections 8.18 to 8.20)."""

import json

import pytest

import mactf
from mactf.management.commands.completion import SHELLS
from mactf.testing.cli import invoke


# -- version ----------------------------------------------------------------------------------

def test_version_text_is_the_bare_version():
    out = invoke(["version"])
    assert out.code == 0 and out.stdout == mactf.__version__ + "\n"


def test_dash_dash_version_and_dash_v_route_to_version():
    assert invoke(["--version"]).stdout == invoke(["-V"]).stdout == mactf.__version__ + "\n"


def test_version_json_keys_and_components_never_import_anything():
    out = invoke(["version", "--format", "json"])
    data = out.json()
    assert {"name", "version", "release", "python", "platform", "install", "project", "components"} <= set(data)
    assert data["version"] == mactf.__version__
    assert set(data["components"]) >= {"torch", "PyYAML", "numpy"}
    assert all(value is None or isinstance(value, str) for value in data["components"].values())


def test_version_components_text_lists_each_dependency():
    out = invoke(["version", "--components"])
    assert out.code == 0
    assert out.stdout.startswith(f"mactf {mactf.__version__}") and "torch" in out.stdout


# -- help -------------------------------------------------------------------------------------

def test_bare_invocation_is_a_usage_error_on_stderr():
    out = invoke([])
    assert out.code == 2 and out.stdout == "" and "usage:" in out.stderr


def test_help_overview_groups_commands_by_category():
    out = invoke(["help"])
    assert out.code == 0
    for category in ("Training", "Runs", "Swarm", "Project", "Scaffold", "Meta"):
        assert category in out.stdout
    for command in ("config", "rules", "peers", "runs", "checkpoints", "logs", "doctor"):
        assert command in out.stdout


def test_dash_h_prints_the_same_as_help():
    assert invoke(["-h"]).stdout == invoke(["--help"]).stdout == invoke(["help"]).stdout


def test_help_for_a_command_and_a_verb():
    assert "usage: mactf runs" in invoke(["help", "runs"]).stdout
    verb = invoke(["help", "runs", "list"]).stdout
    assert "usage: mactf runs list" in verb and "--status" in verb


def test_help_all_lists_the_verbs():
    out = invoke(["help", "--all"]).stdout
    assert "runs list" in out and "config validate" in out


def test_unknown_command_suggests_a_close_one():
    out = invoke(["help", "runz"])
    assert out.code == 2 and "runs" in out.stderr


def test_unknown_subcommand_is_usage_with_the_verbs():
    out = invoke(["help", "runs", "bogus"])
    assert out.code == 2 and "list" in out.stderr


def test_help_json_describes_every_command_with_stable_keys():
    data = invoke(["help", "--format", "json"]).json()
    assert data["schema"] == 1 and data["program"] == "mactf"
    names = [command["name"] for command in data["commands"]]
    assert len(names) == len(set(names))
    for command in data["commands"]:
        assert {"name", "summary", "description", "category", "requires_settings", "destructive", "formats",
                "examples", "exclusive", "arguments", "subcommands"} <= set(command)
        for arg in command["arguments"]:
            assert {"dest", "flags", "kind", "help", "choices", "default", "required", "repeatable",
                    "nargs", "completer", "is_global"} <= set(arg)


def test_help_json_marks_global_options():
    data = invoke(["help", "runs", "list", "--format", "json"]).json()
    flags = {flag: arg["is_global"] for arg in data["arguments"] for flag in arg["flags"]}
    assert flags["--format"] is True and flags["--status"] is False


# -- completion ---------------------------------------------------------------------------------

@pytest.mark.parametrize("shell", SHELLS)
def test_each_shell_renders_a_script_with_an_install_comment(shell):
    out = invoke(["completion", shell])
    assert out.code == 0 and out.stderr == ""
    assert out.stdout.lstrip().startswith(("#", "#!")) or "Register-ArgumentCompleter" in out.stdout
    assert "__complete" in out.stdout


def test_unknown_shell_is_a_usage_error():
    assert invoke(["completion", "tcsh"]).code == 2


@pytest.mark.parametrize("prog", ["bad name", "a;b", "$x", "-x", "a'b"])
def test_hostile_program_names_are_rejected(prog):
    assert invoke(["completion", "bash", "--prog", prog]).code == 2


def test_prog_is_used_in_the_script():
    assert "manage.py" in invoke(["completion", "bash", "--prog", "manage.py"]).stdout


def test_scripts_use_unix_line_endings_on_every_platform():
    from mactf.testing.cli import invoke as run

    assert "\r" not in run(["completion", "powershell"]).stdout


def test_complete_protocol_answers_with_a_directive_first_line():
    out = invoke(["__complete", "fish", "--", "mactf", "runs", "l"])
    assert out.code == 0 and out.stderr == ""
    lines = out.stdout.splitlines()
    assert lines[0].startswith(":") and lines[1].split("\t")[0] == "list"


def test_complete_never_fails_on_garbage():
    for words in (["--"], ["bash", "--", "mactf", "\x00\x01"], ["nope"], []):
        out = invoke(["__complete", *words])
        assert out.code == 0 and out.stderr == ""


def test_complete_offers_choices_and_flags():
    flags = invoke(["__complete", "bash", "--", "mactf", "run", "--form"]).stdout.splitlines()
    assert flags[1:] == ["--format"]
    formats = invoke(["__complete", "bash", "--", "mactf", "rules", "list", "--format", ""]).stdout.splitlines()
    assert set(formats[1:]) == {"text", "json", "yaml"}


def test_complete_never_offers_a_settings_value(project):
    out = project.invoke("__complete", "bash", "--", "mactf", "config", "get", "API.")
    assert out.code == 0
    assert "alice" not in out.stdout and "=" not in "".join(out.stdout.splitlines()[1:])


def test_file_completion_is_a_directive_not_candidates():
    out = invoke(["__complete", "bash", "--", "mactf", "peers", "weights", "--scores-file", ""])
    assert out.stdout.splitlines() == [":1"]
