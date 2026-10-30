"""Import-graph gates: ``mactf.distill`` must stay below the HTTP, client and orchestration layers."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "mactf" / "distill"
HTTP_SIDE = ("mactf.http", "mactf.contrib.api", "mactf.client")
FORBIDDEN_SOURCE = HTTP_SIDE + ("mactf.orchestration",)


def run_isolated(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


def test_no_distill_source_imports_a_forbidden_layer():
    offenders = []
    for path in sorted(PACKAGE.glob("*.py")):
        for line, name in imported_modules_generic(path):
            if any(name == layer or name.startswith(layer + ".") for layer in FORBIDDEN_SOURCE):
                offenders.append(f"{path.name}:{line}: {name}")
    assert not offenders, "\n".join(offenders)


def test_the_resolver_sees_relative_imports():
    """A guard on the guard: the AST walk must resolve ``from ..x`` to ``mactf.x``."""
    names = {name for _, name in imported_modules_generic(PACKAGE / "step.py")}
    assert "mactf.training.optimizer" in names
    assert "mactf.coopetition.weights" in names


@pytest.mark.parametrize("target", ["mactf.distill", "mactf.distill.step", "mactf.distill.config"])
def test_http_side_layers_are_never_loaded(target):
    last = run_isolated(
        f"""
        import sys
        import {target}
        bad = sorted(
            name for name in sys.modules
            if any(name == f or name.startswith(f + ".") for f in {HTTP_SIDE!r})
        )
        print("BAD:" + ",".join(bad))
        """
    )
    assert last == "BAD:"


def test_orchestration_arrives_only_through_the_training_package():
    """If ``mactf.orchestration`` is loaded at all, ``mactf.training`` is the reason."""
    last = run_isolated(
        """
        import sys
        import mactf.distill.step
        orchestration = any(name.startswith("mactf.orchestration") for name in sys.modules)
        training = "mactf.training" in sys.modules
        print(f"IMPLIES:{(not orchestration) or training}")
        """
    )
    assert last == "IMPLIES:True"

def test_transformers_is_not_imported_until_a_transformers_teacher_loads():
    assert (
        run_isolated(
            """
            import sys
            import mactf.distill
            import mactf.distill.step
            print("TRANSFORMERS:" + str("transformers" in sys.modules))
            """
        )
        == "TRANSFORMERS:False"
    )


def test_the_capability_probe_needs_no_import_of_the_step_module():
    assert (
        run_isolated(
            """
            import importlib.util
            import sys
            assert importlib.util.find_spec("mactf.distill") is not None
            print("PROBE:" + str("mactf.distill.step" in sys.modules))
            """
        )
        == "PROBE:False"
    )


def test_rl_functions_do_not_import_distill():
    functions = ROOT / "mactf" / "rl"
    offenders = []
    for path in sorted(functions.rglob("*.py")):
        for line, name in imported_modules_generic(path):
            if name == "mactf.distill" or name.startswith("mactf.distill."):
                offenders.append(f"{path.name}:{line}: {name}")
    assert not offenders, "\n".join(offenders)


def imported_modules_generic(path: Path) -> list[tuple[int, str]]:
    """Imports of any module under ``mactf``, with relative levels resolved from its path."""
    relative = path.relative_to(ROOT).with_suffix("")
    package = list(relative.parts[:-1])
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                name = ".".join(base + ([node.module] if node.module else []))
            else:
                name = node.module or ""
            found.append((node.lineno, name))
    return found
