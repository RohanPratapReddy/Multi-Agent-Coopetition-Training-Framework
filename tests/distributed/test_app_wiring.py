"""The distributed app is optional: no app, no routes; and its modules keep their layering."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import mactf.distributed
from dist_support import configure, teardown
from mactf import checks
from mactf.http.exceptions import Resolver404
from mactf.http.urls import NoReverseMatch, resolve, reverse

PACKAGE = Path(mactf.distributed.__file__).parent
WORKER_NAMES = ["register", "heartbeat", "tasks", "results", "exchange", "deregister"]


def test_routes_exist_only_when_the_app_is_installed(settings):
    configure(settings, distributed=True)
    try:
        for name in WORKER_NAMES:
            assert reverse(f"worker:{name}") == f"/worker/v1/{name}"
            assert resolve(f"/worker/v1/{name}").url_name == name
    finally:
        teardown()

    configure(settings, distributed=False)
    try:
        for name in WORKER_NAMES:
            with pytest.raises(NoReverseMatch):
                reverse(f"worker:{name}")
        with pytest.raises(Resolver404):
            resolve("/worker/v1/register")
        api_names = {entry.name for entry in __import__("mactf.contrib.api.urls", fromlist=["urlpatterns"]).urlpatterns}
        assert not any(str(name).startswith("worker") for name in api_names)
    finally:
        teardown()


def test_base_path_is_configurable(settings):
    configure(settings, distributed=True)
    try:
        settings["DISTRIBUTED"] = {**settings["DISTRIBUTED"], "BASE_PATH": "w/v2"}
        from mactf.http import urls

        urls.clear_url_caches()
        assert reverse("worker:tasks") == "/w/v2/tasks"
    finally:
        teardown()


def test_checks_pass_for_defaults_and_flag_bad_values(settings):
    configure(settings, distributed=True)
    try:
        messages = checks.run_checks() if hasattr(checks, "run_checks") else None
        ids = {m.id for m in messages.messages} if messages is not None else set()
        assert "mactf.distributed.E001" not in ids
        assert "mactf.distributed.W001" not in ids  # the fixture grants a worker token
        assert "mactf.api.E004" not in ids  # 'worker' is a valid scope

        settings["DISTRIBUTED"] = {**settings["DISTRIBUTED"], "WORKER_HEARTBEAT_TIMEOUT": -1, "MAX_WORKERS": 0}
        bad = {m.id: m.message for m in checks.run_checks().messages}
        assert "mactf.distributed.E001" in bad
    finally:
        teardown()


def test_missing_worker_token_warns(settings):
    configure(settings, distributed=True)
    try:
        settings["API"] = {**settings["API"], "TOKENS": {"x": ["read"]}}
        assert "mactf.distributed.W001" in {m.id for m in checks.run_checks().messages}
    finally:
        teardown()


def test_importing_the_package_and_protocol_loads_no_torch_or_http():
    code = (
        "import sys, mactf.distributed, mactf.distributed.protocol\n"
        "bad = [m for m in sys.modules if m == 'torch' or m.startswith(('mactf.http', 'mactf.contrib'))]\n"
        "print(','.join(sorted(bad)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_only_views_and_urls_import_the_http_server_side():
    for source in PACKAGE.glob("*.py"):
        if source.name in {"views.py", "urls.py", "__init__.py"}:
            continue
        for node in ast.parse(source.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ImportFrom) and node.level >= 2:
                head = (node.module or "").split(".")[0]
                assert head not in {"http", "contrib"}, f"{source.name} imports mactf.{node.module} at module scope"
