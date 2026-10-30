"""Criteria 43 and 44: ``mactf/http`` is independent of the engine and of the API app."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import mactf.http as http_package

ENGINE = (
    "orchestration", "training", "scoring", "coopetition", "rules", "sandbox", "contrib",
    "client", "hierarchy", "rl", "data", "distributed", "checkpoints", "models",
)


def _run(code: str) -> str:
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=90)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_no_http_module_imports_an_engine_module():
    root = Path(http_package.__file__).parent
    offenders = []
    for source in root.glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            targets = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level >= 2:  # ``from ..x import`` leaves the http package
                    targets = [f"mactf.{base}"] + [f"mactf.{base}.{a.name}" for a in node.names]
                elif node.level == 0:
                    targets = [base]
            for target in targets:
                parts = target.split(".")
                if parts[0] == "mactf" and len(parts) > 1 and parts[1] in ENGINE:
                    offenders.append(f"{source.name}: {target}")
    assert offenders == []


def test_importing_http_loads_no_engine_module():
    out = _run(
        f"""
        import sys
        import mactf.http
        banned = {ENGINE!r}
        print([m for m in sys.modules if m.startswith("mactf.") and m.split(".")[1] in banned])
        """
    )
    assert out == "[]"


def test_http_is_importable_and_serves_with_the_api_app_blocked():
    out = _run(
        """
        import sys
        sys.modules["mactf.contrib.api"] = None
        sys.modules["mactf.contrib"] = None
        import mactf.http
        from mactf.http import path, View, JsonResponse
        from mactf.http.handler import WSGIHandler
        from mactf.http.testing import dispatch_wsgi
        import types

        def ping(request):
            return JsonResponse({"ok": True})

        urlconf = types.ModuleType("u")
        urlconf.urlpatterns = [path("ping/", ping, name="ping")]
        print(dispatch_wsgi(WSGIHandler(urlconf=urlconf), "GET", "/ping/").status_code)
        """
    )
    assert out == "200"


def test_the_http_suite_does_not_need_the_app(tmp_path):
    """The shared fixtures import only ``mactf.http``; the app-dependent fixture is separate."""
    fixtures = Path(__file__).with_name("http_fixtures.py").read_text(encoding="utf-8")
    assert "mactf.contrib" not in fixtures and "mactf.client" not in fixtures
