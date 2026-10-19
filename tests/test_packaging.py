"""Packaging guarantees: version source, default settings, scaffolder, metadata."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_version_matches_package():
    from mactf import __version__

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["dynamic"] == ["version"]
    assert data["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "mactf.__version__"

    # What the build backend will resolve.
    import mactf

    assert data["project"]["name"] == "mactf"
    assert mactf.__version__ == __version__
    assert mactf.VERSION == tuple(int(p) for p in __version__.split("."))


def test_pyproject_declares_runtime_contract():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requires = " ".join(data["project"]["dependencies"])
    assert "torch" in requires
    assert "PyYAML" in requires
    extras = data["project"]["optional-dependencies"]
    for name in ("gpu", "sandbox", "shell", "dev", "docs"):
        assert name in extras, name
    assert data["project"]["scripts"]["mactf"] == "mactf.management.cli:main"
    assert data["project"]["license"] == "MIT"
    assert data["project"]["requires-python"] == ">=3.10"


def test_default_settings_load_without_a_project():
    from mactf import conf

    conf.reset()
    loaded = conf.load_settings("mactf.settings_default")
    try:
        assert loaded["PEERS"] == []
        assert loaded["CONFIG_FILES"] == []
        assert all(app.startswith("mactf.contrib.") for app in loaded["INSTALLED_APPS"])
        # Validation must have run: the non-offsettable floors are intact.
        assert {"P1", "P2", "P17", "P20", "P21", "P22"} <= set(
            loaded["SCORING"]["NON_OFFSETTABLE"]
        )
    finally:
        conf.reset()


def test_default_settings_module_is_the_fallback():
    from mactf import conf

    assert conf.DEFAULT_SETTINGS_MODULE == "mactf.settings_default"


def test_cli_entry_point_defaults_settings_module(monkeypatch):
    from mactf.management import cli

    monkeypatch.delenv("MACTF_SETTINGS_MODULE", raising=False)
    captured: dict[str, list[str]] = {}

    def fake_execute(argv=None):
        captured["argv"] = list(argv or [])
        return 0

    import mactf.management

    monkeypatch.setattr(mactf.management, "execute_from_command_line", fake_execute)
    assert cli.main(["version"]) == 0
    assert captured["argv"] == ["mactf", "version"]
    assert __import__("os").environ["MACTF_SETTINGS_MODULE"] == "mactf.settings_default"


def test_startproject_writes_runnable_project(tmp_path, monkeypatch):
    from mactf import conf
    from mactf.management.commands import startproject

    written = startproject.write_project("demo_swarm", target_root=tmp_path)
    names = {path.relative_to(tmp_path).as_posix() for path in written}
    assert names == {
        "demo_swarm/manage.py",
        "demo_swarm/__init__.py",
        "demo_swarm/settings.py",
        "demo_swarm/peers/__init__.py",
        "demo_swarm/configs/swarm.yaml",
        "demo_swarm/configs/scoring.yaml",
        "demo_swarm/configs/sandbox.yaml",
        "demo_swarm/.gitignore",
    }

    settings_text = (tmp_path / "demo_swarm" / "settings.py").read_text(encoding="utf-8")
    assert "{{ project_name }}" not in settings_text
    assert "demo_swarm.peers" in settings_text
    # Braces must survive substitution untouched.
    assert "PEERS = [" in settings_text

    monkeypatch.syspath_prepend(str(tmp_path))
    for module in [m for m in sys.modules if m == "demo_swarm" or m.startswith("demo_swarm.")]:
        monkeypatch.delitem(sys.modules, module, raising=False)

    conf.reset()
    loaded = conf.load_settings("demo_swarm.settings")
    try:
        assert len(loaded["PEERS"]) == 3
        assert loaded["PATHS"]["ROOT"] == str((tmp_path / "demo_swarm").resolve())
        for entry in loaded["CONFIG_FILES"]:
            assert (tmp_path / "demo_swarm" / entry).is_file(), entry
    finally:
        conf.reset()


def test_startproject_refuses_to_overwrite(tmp_path):
    from mactf.management.commands import startproject
    from mactf.exceptions import CommandError

    startproject.write_project("demo_swarm", target_root=tmp_path)
    with pytest.raises(CommandError):
        startproject.write_project("demo_swarm", target_root=tmp_path)
    startproject.write_project("demo_swarm", target_root=tmp_path, force=True)


@pytest.mark.parametrize("name", ["", "1bad", "has-dash", "class", "with space"])
def test_startproject_rejects_invalid_names(name, tmp_path):
    from mactf.management.commands import startproject
    from mactf.exceptions import CommandError

    with pytest.raises(CommandError):
        startproject.write_project(name, target_root=tmp_path)


def test_templates_are_installed_package_data():
    from mactf.management.commands import startproject

    root = startproject.template_root()
    for template_name, _ in startproject.LAYOUT:
        assert (root / template_name).is_file(), template_name


def test_training_manage_py_still_targets_its_own_settings():
    text = (ROOT / "training" / "manage.py").read_text(encoding="utf-8")
    assert '"training.settings"' in text
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "pip install mactf" in readme
    assert "twine upload" in readme
