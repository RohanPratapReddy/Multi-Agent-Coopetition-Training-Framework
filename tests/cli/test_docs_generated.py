import importlib.util
from pathlib import Path

import pytest

from mactf.management.exitcodes import ExitCode

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_cli_docs", ROOT / "scripts" / "gen_cli_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_reference_is_what_the_generator_produces(gen):
    assert gen.committed() == gen.generate(), "docs/cli.md is stale: python scripts/gen_cli_docs.py"


def test_generation_does_not_depend_on_the_working_directory(gen, tmp_path, monkeypatch):
    first = gen.generate()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert gen.generate() == first


def test_every_command_and_verb_has_a_section(gen):
    data = gen.payload()
    text = gen.committed()
    assert "serve" in {command["name"] for command in data["commands"]}
    for command in data["commands"]:
        assert f"## `mactf {command['name']}`" in text, command["name"]
        for sub in command["subcommands"]:
            assert f"### `mactf {command['name']} {sub['name']}`" in text, (command["name"], sub["name"])


def test_every_exit_code_is_described(gen):
    assert set(gen.EXIT_MEANING) == {code.name for code in ExitCode}


def test_the_reference_holds_no_machine_specific_text(gen):
    text = gen.committed()
    assert str(ROOT) not in text and str(Path.home()) not in text


def test_check_mode_reports_a_stale_file(gen, tmp_path, monkeypatch, capsys):
    stale = tmp_path / "cli.md"
    stale.write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(gen, "TARGET", stale)
    assert gen.main(["--check"]) == 1
    assert "stale" in capsys.readouterr().err
    assert stale.read_text(encoding="utf-8") == "old\n"          # --check never writes
