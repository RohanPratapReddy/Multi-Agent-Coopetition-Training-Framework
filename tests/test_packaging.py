"""Packaging guarantees: version source, default settings, scaffolder, metadata."""

from __future__ import annotations

import re
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 has no stdlib tomllib.
    import tomli as tomllib

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
    # VERSION is the numeric release segment, so pre-release suffixes are dropped
    # rather than raising on int("0rc1").
    assert mactf.VERSION == tuple(
        int(part) for part in re.match(r"\d+(?:\.\d+)*", __version__).group(0).split(".")
    )


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
    from mactf.conf_files import CONF_LAYOUT

    assert names == {
        "demo_swarm/manage.py",
        "demo_swarm/__init__.py",
        "demo_swarm/settings.py",
        "demo_swarm/peers/__init__.py",
        *(f"demo_swarm/conf/{entry.filename}" for entry in CONF_LAYOUT),
        "demo_swarm/.gitignore",
    }
    assert not (tmp_path / "demo_swarm" / "configs").exists()

    settings_text = (tmp_path / "demo_swarm" / "settings.py").read_text(encoding="utf-8")
    assert "{{ project_name }}" not in settings_text
    assert "demo_swarm.peers" in settings_text
    # settings.py is identity only; the swarm lives in conf/peers.yaml.
    assert "PEERS" not in settings_text
    assert "CONF_DIR" in settings_text

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

    for template_name, _ in startproject.LAYOUT:
        source, _package = startproject.template_source(template_name)
        assert source.is_file(), template_name


def test_training_manage_py_still_targets_its_own_settings():
    text = (ROOT / "training" / "manage.py").read_text(encoding="utf-8")
    assert '"training.settings"' in text
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "pip install mactf" in readme
    assert "twine upload" in readme


# -- extras, wheel contents and optional backends (document 06 section 4) --------------
KNOWN_EXTRAS = {
    "gpu", "sandbox", "shell", "asgi", "scipy", "peft", "quant", "train", "tensorboard", "wandb", "docs", "dev",
}


def _project():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def _names(requirements):
    return {re.split(r"[<>=!~\[; ]", item, maxsplit=1)[0].lower() for item in requirements}


def test_extras_are_exactly_the_documented_set():
    assert set(_project()["optional-dependencies"]) == KNOWN_EXTRAS


@pytest.mark.parametrize("gone", ["api", "sdk", "server"])
def test_the_service_and_client_are_core_so_there_is_no_opt_in_extra(gone):
    assert gone not in _project()["optional-dependencies"]


def test_heavy_optional_backends_stay_out_of_the_base_dependencies():
    base = _names(_project()["dependencies"])
    assert base == {"torch", "pyyaml"}
    assert not base & {"scipy", "peft", "bitsandbytes", "tensorboard", "wandb", "transformers", "uvicorn"}


def test_train_contains_the_scipy_and_peft_dependencies_but_not_bitsandbytes():
    extras = _project()["optional-dependencies"]
    train = _names(extras["train"])
    assert _names(extras["scipy"]) <= train and _names(extras["peft"]) <= train
    assert "bitsandbytes" not in train and _names(extras["quant"]) == {"bitsandbytes"}


def test_dev_installs_everything_the_full_suite_needs():
    dev = _names(_project()["optional-dependencies"]["dev"])
    assert {"pytest", "uvicorn", "scipy", "peft", "transformers", "build", "twine"} <= dev


def test_uvicorn_is_imported_only_by_the_asgi_adapter():
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "mactf").rglob("*.py")
        if re.search(r"^\s*(import|from)\s+uvicorn\b", path.read_text(encoding="utf-8"), re.M)
        and path.relative_to(ROOT / "mactf").as_posix() != "http/asgi.py"
    ]
    assert offenders == []


def test_release_version_is_the_pre_production_candidate():
    """0.0.0rc2 is a pre-release, so the deploy workflow routes it to TestPyPI only."""
    from packaging.version import Version

    from mactf import VERSION, __version__

    assert Version(__version__) == Version("0.0.0rc2") and Version(__version__).is_prerelease
    assert VERSION == (0, 0, 0)


def test_scipy_optimizer_names_the_extra_when_scipy_is_missing(monkeypatch):
    import builtins

    from mactf.exceptions import ImproperlyConfigured
    from mactf.training.optimizer import ScipyOptimizer

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("No module named 'scipy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for cached in [m for m in sys.modules if m == "scipy" or m.startswith("scipy.")]:
        monkeypatch.delitem(sys.modules, cached, raising=False)
    with pytest.raises(ImproperlyConfigured, match=re.escape('pip install "mactf[scipy]"')):
        ScipyOptimizer("L-BFGS-B")


def test_the_optimizer_module_imports_with_no_extras():
    import subprocess

    code = (
        "import sys\nfor n in ('scipy','peft','transformers','bitsandbytes','tensorboard','wandb','uvicorn'):\n"
        "    sys.modules[n] = None\nimport mactf.training.optimizer\nprint('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


@pytest.fixture(scope="module")
def wheel_names(tmp_path_factory):
    """Build the wheel from a clean copy of the sources, the way ``python -m build`` would."""
    import shutil
    import subprocess
    import zipfile

    work = tmp_path_factory.mktemp("wheel")
    source = work / "src"
    source.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE", "MANIFEST.in"):
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, source / name)
    shutil.copytree(ROOT / "mactf", source / "mactf", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    out = work / "dist"
    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation", "-w", str(out), "-q"],
        cwd=source, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        pytest.skip(f"cannot build a wheel offline here: {result.stderr[-300:]}")
    wheel = next(out.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        return set(archive.namelist())


@pytest.mark.parametrize(
    "package",
    [
        "mactf/http", "mactf/contrib/api", "mactf/contrib/api/views", "mactf/client", "mactf/testing",
        "mactf/data", "mactf/distributed", "mactf/training", "mactf/rl/functions",
        "mactf/scoring", "mactf/sandbox",
    ],
)
def test_wheel_contains_every_new_package(wheel_names, package):
    assert f"{package}/__init__.py" in wheel_names, package


def test_wheel_ships_py_typed_and_templates_and_no_tests(wheel_names):
    assert "mactf/py.typed" in wheel_names
    assert any(name.endswith(".tpl") for name in wheel_names)
    assert not any(name.startswith(("tests/", "training/", "docs/")) for name in wheel_names)


def test_wheel_ships_every_conf_template_and_no_legacy_configs(wheel_names):
    from mactf.conf_files import CONF_LAYOUT

    missing = [
        entry.template_name
        for entry in CONF_LAYOUT
        if f"mactf/templates/conf/{entry.template_name}" not in wheel_names
    ]
    assert missing == []
    assert "mactf/templates/conf/__init__.py" in wheel_names
    assert not any(name.startswith("mactf/templates/project/configs/") for name in wheel_names)


def test_every_source_package_directory_is_importable_so_the_wheel_cannot_drop_it():
    missing = []
    for directory in (ROOT / "mactf").rglob("*"):
        if not directory.is_dir() or directory.name in {"__pycache__", "templates", "project", "configs", "peers"}:
            continue
        if any(part in {"__pycache__"} for part in directory.parts):
            continue
        if any(child.suffix == ".py" for child in directory.iterdir()) and not (directory / "__init__.py").exists():
            missing.append(directory.relative_to(ROOT).as_posix())
    assert missing == []


# -- CI gates (document 06 section 5) --------------------------------------------------
def _workflow():
    import yaml

    return yaml.safe_load((ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8"))


def test_workflow_has_every_required_gate_and_build_waits_for_them():
    jobs = _workflow()["jobs"]
    gates = {
        "api-contract", "stdlib-only", "numeric-equivalence", "determinism",
        "points-equivalence", "points-parity", "transport-parity",
    }
    assert gates <= set(jobs)
    assert gates <= set(jobs["build"]["needs"]) and "test" in jobs["build"]["needs"]


def test_workflow_tests_windows_on_python_312_only():
    matrix = _workflow()["jobs"]["test"]["strategy"]["matrix"]
    assert matrix["os"] == ["ubuntu-latest"]
    assert matrix["include"] == [{"os": "windows-latest", "python-version": "3.12"}]
    assert not any("macos" in str(entry) for entry in [matrix])


def test_every_gate_names_test_files_that_exist():
    jobs = _workflow()["jobs"]
    missing = []
    for name, job in jobs.items():
        for step in job.get("steps", []):
            for token in str(step.get("run", "")).split():
                if token.startswith("tests/") and token.endswith(".py") and not (ROOT / token).exists():
                    missing.append((name, token))
    assert missing == []


def test_determinism_gate_runs_under_two_hash_seeds():
    job = _workflow()["jobs"]["determinism"]
    assert job["strategy"]["matrix"]["hashseed"] == ["0", "12345"]
    assert any("PYTHONHASHSEED" in step.get("env", {}) for step in job["steps"])


def test_version_gate_uses_packaging_version_not_a_string_check():
    text = (ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
    assert "packaging.version" in text and "is_prerelease" in text
    from packaging.version import Version

    assert not Version("0.1.0").is_prerelease
    assert all(Version(v).is_prerelease for v in ("0.1.0rc1", "0.1.0a1", "0.1.0b2"))