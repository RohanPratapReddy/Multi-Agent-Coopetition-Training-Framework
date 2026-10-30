"""Criterion 39: ``import mactf.http`` needs no third-party HTTP package."""

from __future__ import annotations

import subprocess
import sys
import textwrap

BANNED = ["fastapi", "starlette", "pydantic", "httpx", "uvicorn", "flask", "django", "requests", "aiohttp", "werkzeug"]


def _run(code: str) -> str:
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_import_loads_no_third_party_http_package():
    out = _run(
        f"""
        import sys
        import mactf.http, mactf.http.openapi, mactf.http.testing, mactf.http.handler
        banned = set({BANNED!r})
        print(sorted(banned & {{m.split('.')[0] for m in sys.modules}}))
        """
    )
    assert out == "[]"


def test_import_works_when_every_http_package_is_blocked():
    out = _run(
        f"""
        import sys
        for name in {BANNED!r}:
            sys.modules[name] = None  # any import of these now raises ImportError
        import mactf.http
        from mactf.http import handler, openapi, testing
        r = testing.dispatch_wsgi(handler.WSGIHandler(urlconf=None), "GET", "/nope/")
        print(r.status_code)
        """
    )
    assert out == "404"


def test_every_http_module_only_imports_stdlib_or_mactf():
    import ast
    from pathlib import Path

    import mactf.http as package

    allowed_third_party: set[str] = set()
    stdlib = set(sys.stdlib_module_names)
    for source in Path(package.__file__).parent.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                assert name in stdlib or name == "mactf" or name in allowed_third_party, (
                    f"{source.name} imports {name!r}, which is not stdlib"
                )
